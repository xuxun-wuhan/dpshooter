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

import android.graphics.Bitmap;
import android.util.Log;
import android.util.Size;
import android.widget.ImageView;

/**
 * Main activity class handling camera access and image processing.
 */
public class MainActivity extends AppCompatActivity {
    /**
     * Used to load the 'native-lib' library on application startup.
     */
    static {
        System.loadLibrary("native-lib");
    }

    static final int imageWidth = 640;
    static final int imageHeight = 480;

    private static final String TAG = "MainActivity";
    private static final int PERMISSION_REQUEST_CAMERA = 0;
    private PreviewView previewView;
    private ImageView opticalFlowView;

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    /**
     * Native method for getting optical flow image.
     *
     * @param prevPic ByteBuffer of the previous image.
     * @param nextPic ByteBuffer of the current image.
     * @param imageWidth Width of the image.
     * @param imageHeight Height of the image.
     * @return Byte array representing the processed image.
     */
    public native byte[] getOpticalFlowImage(ByteBuffer prevPic, ByteBuffer nextPic, int imageWidth, int imageHeight);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.viewFinder);
        opticalFlowView = findViewById(R.id.opticalFlowView);
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        if (allPermissionsGranted()) {
            startCamera(); //start camera if permission has been granted by user
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST_CAMERA);
        }
    }

    /**
     * Check if all required permissions are granted.
     *
     * @return True if all permissions are granted, false otherwise.
     */
    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Start camera view.
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
     * Bind the camera preview to the app.
     *
     * @param cameraProvider ProcessCameraProvider instance.
     */
    private void bindPreview(ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(imageWidth, imageHeight))
                .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), new ImageAnalysis.Analyzer() {
            private ByteBuffer prevMat = null;

            @Override
            public void analyze(ImageProxy image) {
                // TODO: Validate image resolution support and compactness of Y channel (image pitch = image width).
                // Initialize prevMat on first frame
                if (prevMat == null) {
                    prevMat = ByteBuffer.allocateDirect(imageHeight * imageWidth);
                    ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
                    ByteBuffer yBuffer = yPlane.getBuffer();
                    prevMat.rewind();
                    yBuffer.rewind();
                    prevMat.put(yBuffer);
                    image.close();
                    return;
                }
                ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
                ByteBuffer nextMat = yPlane.getBuffer();

                byte[] byteArray = getOpticalFlowImage(prevMat, nextMat, imageWidth, imageHeight);
                Bitmap bitmap = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
                ByteBuffer buffer = ByteBuffer.wrap(byteArray);
                bitmap.copyPixelsFromBuffer(buffer);

                nextMat.rewind();
                prevMat.rewind();
                prevMat.put(nextMat);

                // Display bitmap on screen using ImageView
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        opticalFlowView.setImageBitmap(bitmap);
                    }
                });
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
}
