package com.example.dualphoneshooter;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
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
import android.widget.ImageView;

/**
 * @brief Main activity class handling camera access and image processing.
 */
public class MainActivity extends AppCompatActivity {
    /**
     * @brief Used to load the 'native-lib' library on application startup.
     */
    static {
        System.loadLibrary("native-lib");
    }

    /// @brief Width of the image.
    static final int imageWidth = 640;

    /// @brief Height of the image.
    static final int imageHeight = 480;

    /// @brief Tag for log messages.
    private static final String TAG = "MainActivity";

    /// @brief Request code for camera permission request.
    private static final int PERMISSION_REQUEST_CAMERA = 0;

    /// @brief View for displaying the camera preview.
    private PreviewView previewView;

    /// @brief View for displaying the processed image.
    private ImageView opticalFlowView;

    /// @brief Worker threads for background tasks.
    private HandlerThread[] workThreads;

    /// @brief Handlers associated with worker threads.
    private Handler[] workHandlers;

    /// @brief Future that provides access to the camera provider.
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    /**
     * @brief Native method for getting optical flow image.
     * @param prevImg Buffer array of the previous image.
     * @param nextImg Buffer array of the current image.
     * @param imageWidth Width of the image.
     * @param imageHeight Height of the image.
     * @return Byte array representing the processed image.
     */
    public native byte[] getOpticalFlowImage(byte[] prevImg, byte[] nextImg, int imageWidth, int imageHeight);

    /**
     * @brief Set up the activity.
     * @param savedInstanceState Saved activity state.
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Set up worker threads
        workThreads = new HandlerThread[3];
        workHandlers = new Handler[3];
        for(int i=0; i<3; i++) {
            workThreads[i] = new HandlerThread("work_thread_"+i);
            workThreads[i].start();
            workHandlers[i] = new Handler(workThreads[i].getLooper());
        }

        // Get view references
        previewView = findViewById(R.id.viewFinder);
        opticalFlowView = findViewById(R.id.opticalFlowView);

        // Request camera provider
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        if (allPermissionsGranted()) {
            startCamera(); //start camera if permission has been granted by user
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST_CAMERA);
        }
    }

    /**
     * @brief Check if all required permissions are granted.
     * @return True if all permissions are granted, false otherwise.
     */
    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * @brief Start the camera view.
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
     * @brief Bind the camera preview to the app.
     * @param cameraProvider ProcessCameraProvider instance.
     */
    private void bindPreview(ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(imageWidth, imageHeight))
                .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), new ImageAnalysis.Analyzer() {
            /// @brief Previous image.
            private byte[] prevPic = null;

            /// @brief Current image.
            private byte[] nextPic = null;

            /// @brief Index of the worker thread to be used for the next task.
            private AtomicInteger workerIndex = new AtomicInteger(0);  // Thread-safe counter

            /**
             * @brief Analyze an image to process it and update the UI.
             * @param image The image to be analyzed.
             */
            @Override
            public void analyze(ImageProxy image) {
                // TODO: Validate image resolution support and compactness of Y channel (image pitch = image width).
                // Initialize prevMat on first frame
                if (prevPic == null) {
                    ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
                    ByteBuffer yBuffer = yPlane.getBuffer();
                    yBuffer.rewind();
                    prevPic = new byte[yBuffer.remaining()];
                    yBuffer.get(prevPic);
                    image.close();
                    return;
                }
                ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
                ByteBuffer yBuffer = yPlane.getBuffer();
                yBuffer.rewind();
                nextPic = new byte[yBuffer.remaining()];
                yBuffer.get(nextPic);

                // Copy the current image pairs to avoid thread safety issues
                byte[] localPrevPic = prevPic.clone();

                // Schedule the task on a worker thread, rotating between them
                int workerId = workerIndex.getAndUpdate(i -> (i + 1) % workHandlers.length);
                workHandlers[workerId].post(() -> processImagePairs(localPrevPic, nextPic, imageWidth, imageHeight));

                prevPic = nextPic;
                image.close();
            }
        });

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        try {
            // Unbind use cases before rebinding
            cameraProvider.unbindAll();

            // Bind use cases to camera
            Camera camera = cameraProvider.bindToLifecycle(
                    (LifecycleOwner) this, cameraSelector, preview, imageAnalysis);

        } catch (Exception exc) {
            Log.e(TAG, "Use case binding failed", exc);
        }
    }

    /**
     * @brief Process a pair of images and update the UI.
     * @param prevImg Buffer array of the previous image.
     * @param nextImg Buffer array of the current image.
     * @param w Width of the images.
     * @param h Height of the images.
     */
    void processImagePairs(byte[] prevImg, byte[] nextImg, int w, int h ) {
        byte[] byteArray = getOpticalFlowImage(prevImg, nextImg, w, h);
        Bitmap bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        ByteBuffer buffer = ByteBuffer.wrap(byteArray);
        bitmap.copyPixelsFromBuffer(buffer);

        // Display bitmap on screen using ImageView
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                opticalFlowView.setImageBitmap(bitmap);
            }
        });
    }
}
