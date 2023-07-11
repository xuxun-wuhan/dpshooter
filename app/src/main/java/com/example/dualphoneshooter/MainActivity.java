package com.example.dualphoneshooter;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;

import com.google.common.util.concurrent.ListenableFuture;

import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;

import android.graphics.Bitmap;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.SurfaceView;
import android.widget.ImageView;

/**
 * MainActivity class is the primary entry point of the application.
 * This class sets up the application, initializes the camera and manages the image processing.
 */
public class MainActivity extends AppCompatActivity {
    static final int imageWidth = 640;
    static final int imageHeight = 480;
    private static final String TAG = "MainActivity";
    private static final int PERMISSION_REQUEST_CAMERA = 0;

    /**
     * Load native libraries on startup.
     */
    static {
        System.loadLibrary("native-lib");
    }

    private OpenGLProcessor openGLProcessor;
    private SurfaceView previewView;
    private ImageView opticalFlowView;
    private HandlerThread[] workThreads;
    private Handler[] workHandlers;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    /**
     * Method to retrieve optical flow image data.
     *
     * @param prevImg             Previous image data.
     * @param nextImg             Next image data.
     * @param opticalFlowVertices The OpenGL vertices used for optical flow based forward warping.
     * @param imageWidth          Width of the images.
     * @param imageHeight         Height of the images.
     * @return Processed image data.
     */
    private native byte[] getOpticalFlowImage(byte[] prevImg, byte[] nextImg, float[] opticalFlowVertices, int imageWidth, int imageHeight);

    /**
     * Called when the activity is starting.
     *
     * @param savedInstanceState If the activity is being re-initialized after previously being shut down then this Bundle contains the data it most recently supplied.
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        workThreads = new HandlerThread[3];
        workHandlers = new Handler[3];
        for (int i = 0; i < 3; i++) {
            workThreads[i] = new HandlerThread("work_thread_" + i);
            workThreads[i].start();
            workHandlers[i] = new Handler(workThreads[i].getLooper());
        }

        openGLProcessor = new OpenGLProcessor(imageWidth, imageHeight);

        previewView = findViewById(R.id.viewFinder);
        opticalFlowView = findViewById(R.id.opticalFlowView);

        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST_CAMERA);
        }
    }

    /**
     * Perform any final cleanup before an activity is destroyed.
     */
    @Override
    protected void onDestroy() {
        super.onDestroy();

        // TODO: add openGLProcessor cleanup code if necessary
    }

    /**
     * Check if all the required permissions are granted.
     *
     * @return True if permissions are granted, false otherwise.
     */
    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Starts the camera and binds the use cases.
     */
    private void startCamera() {
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                // No errors need to be handled for this Future.
                // This should never be reached.
            }
        }, ContextCompat.getMainExecutor(this));
    }

    /**
     * Bind camera data to our Preview, along with analysis use cases.
     *
     * @param cameraProvider Camera provider used to bind the lifecycle, use cases and camera selector.
     */
    private void bindPreview(ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        // Set up a Surface tied to the previewView in the OpenGLProcessor
        Surface previewViewSurface = previewView.getHolder().getSurface();
        int previewWidth = previewView.getHolder().getSurfaceFrame().width();
        int previewHeight = previewView.getHolder().getSurfaceFrame().height();
        openGLProcessor.setOutgoingSurface(previewViewSurface, previewWidth, previewHeight);

        // Create a surface with the camera texture and provide it to the preview
        Surface cameraTextureSurface = new Surface(openGLProcessor.getCameraTexture());
        preview.setSurfaceProvider(request -> request.provideSurface(cameraTextureSurface, ContextCompat.getMainExecutor(this), result -> {
        }));

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(imageWidth, imageHeight))
                .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), new ImageAnalysis.Analyzer() {
            private byte[] prevPic = null;
            private byte[] nextPic = null;
            private long prevPicTimestamp = 0;
            private long nextPicTimestamp = 0;
            private AtomicInteger workerIndex = new AtomicInteger(0);  // Thread-safe counter

            @Override
            public void analyze(ImageProxy image) {
                if (prevPic == null) {
                    ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
                    ByteBuffer yBuffer = yPlane.getBuffer();
                    yBuffer.rewind();
                    prevPic = new byte[yBuffer.remaining()];
                    yBuffer.get(prevPic);
                    prevPicTimestamp = image.getImageInfo().getTimestamp();
                    image.close();
                    return;
                }
                ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
                ByteBuffer yBuffer = yPlane.getBuffer();
                yBuffer.rewind();
                nextPic = new byte[yBuffer.remaining()];
                yBuffer.get(nextPic);
                nextPicTimestamp = image.getImageInfo().getTimestamp();

                byte[] localPrevPic = prevPic.clone();
                long localPrevPicTimestamp = prevPicTimestamp;

                int workerId = workerIndex.getAndUpdate(i -> (i + 1) % workHandlers.length);
                workHandlers[workerId].post(() -> processImagePairs(localPrevPic, nextPic, localPrevPicTimestamp, imageWidth, imageHeight));

                prevPic = nextPic;
                prevPicTimestamp = nextPicTimestamp;
                image.close();
            }
        });

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        try {
            cameraProvider.unbindAll();

            Camera camera = cameraProvider.bindToLifecycle(
                    (LifecycleOwner) this, cameraSelector, preview, imageAnalysis);

        } catch (Exception exc) {
            Log.e(TAG, "Use case binding failed", exc);
        }
    }

    /**
     * Process an image pair to generate and display an optical flow image.
     *
     * @param prevImg Previous image data.
     * @param nextImg Next image data.
     * @param w       Width of the images.
     * @param h       Height of the images.
     */
    void processImagePairs(byte[] prevImg, byte[] nextImg, long prevImgTimestamp, int w, int h) {
        int glWidth = w / 2;
        int glHeight = h / 2;
        float[] opticalFlowVertices = new float[glWidth * glHeight * 2];
        byte[] byteArray = getOpticalFlowImage(prevImg, nextImg, opticalFlowVertices, w, h);
        //Submit optical flow estimation and frm0's timestamp for forward warping
        openGLProcessor.addOpticalFlowVertices(prevImgTimestamp, opticalFlowVertices);
        Bitmap bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        ByteBuffer buffer = ByteBuffer.wrap(byteArray);
        bitmap.copyPixelsFromBuffer(buffer);

        runOnUiThread(() -> opticalFlowView.setImageBitmap(bitmap));
    }
}
