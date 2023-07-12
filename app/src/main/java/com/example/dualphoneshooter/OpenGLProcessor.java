package com.example.dualphoneshooter;

import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGL14;

import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.view.Surface;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.concurrent.PriorityBlockingQueue;

/**
 * @class OpenGLProcessor
 * @brief This class handles OpenGL related operations for DualPhoneShooter.
 *
 * The class is responsible for handling the OpenGL rendering process, and it includes setting up
 * the OpenGL context, rendering frames from the camera to a Framebuffer Object (FBO), and rendering
 * frames from the FBO to the screen.
 */
public class OpenGLProcessor {
    // Static configuration variables
    private static final String TAG = "OpenGLProcessor";
    private static final int FBO_COUNT = 8;
    private final int width;
    private final int height;

    private final OpenGLUtils.Framebuffer[] fboRingBuffer = new OpenGLUtils.Framebuffer[FBO_COUNT];
    // A queue that's sorted by timestamp.
    // This queue stores the vertices of optical flow, which is used for rendering from FBO to screen.
    PriorityBlockingQueue<OpticalFlowVertices> OpticalFlowVertexBuffer = new PriorityBlockingQueue<>();
    private EGLContext eglContext;
    // Rectangle vertices, these coordinates are used for drawing the camera frames onto FBO.
    private float squareCoords[] = {
            -1.0f, 1.0f,   // top left
            -1.0f, -1.0f,  // bottom left
            1.0f, 1.0f,    // top right
            1.0f, -1.0f}; // bottom right

    // Texture coordinates, these coordinates are used for mapping the camera frames onto the rectangle vertices.
    // Note: In OpenGL, texture coordinates are specified with (0, 0) at the bottom-left corner,
    // and (1, 1) at the top-right. However, the SurfaceTexture from the camera has (0, 0)
    // at the top-left and (1, 1) at the bottom-right, which causes the upside-down image.
    private float textureCoords[] = {
            0.0f, 0.0f,  // top left
            0.0f, 1.0f,  // bottom left
            1.0f, 0.0f,  // top right
            1.0f, 1.0f}; // bottom right

    // Vertex buffers, Texture buffers, and Index buffers
    // These buffers are used for storing the vertex and texture data and passing them to the OpenGL Shaders.
    private FloatBuffer vertexBuffer;
    private FloatBuffer textureBuffer;
    private FloatBuffer vertexBufferForFBO;
    private FloatBuffer textureBufferForFBO;
    private IntBuffer indexBufferForFBO;
    private int indexBufferId;
    private int currentFBOIndex = 0;
    private int textureId;
    private SurfaceTexture cameraTexture;
    private Surface outgoingSurface;
    private int outgoingSurfaceWidth;
    private int outgoingSurfaceHeight;
    private int vertexShader;
    private int fragmentShader;
    private int shaderProgram;
    // Handles for passing in data to the shaders
    private int positionHandle;
    private int textureCoordHandle;
    private int fragmentShader2;
    private int shaderProgram2;
    private int positionHandle2;
    private int textureCoordHandle2;
    private boolean isInitialized;

    /**
     * @param width  Width of the OpenGL context.
     * @param height Height of the OpenGL context.
     * @brief Constructor for OpenGLProcessor.
     *
     * The constructor initializes the width and height of the OpenGL context.
     */
    public OpenGLProcessor(int width, int height) {
        // Initialize dimensions
        this.width = width;
        this.height = height;
        isInitialized = false;
    }

    /**
     * @brief Native method for getting texture coordinates.
     *
     * This method is implemented in the native library and it fills the buffer with texture coordinates.
     */
    private native void getTextureCoords(FloatBuffer buffer, int w, int h);

    /**
     * @brief Native method for getting triangle index buffer.
     *
     * This method is implemented in the native library and it fills the buffer with the indices of the vertices of the triangles.
     */
    private native void getTriangleIndexBuffer(IntBuffer buffer, int w, int h);

    /**
     * @brief Init code. Called only after outgoingSurface is available.
     *
     * This method initializes the OpenGL environment, which includes creating a SurfaceTexture, setting up the EGL display, creating framebuffers, setting up vertex data, setting up shaders and programs, and generating index buffer for OpenGL.
     */
    private void initialize() {
        // Generate texture for SurfaceTexture
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        textureId = textures[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);

        // Create SurfaceTexture
        cameraTexture = new SurfaceTexture(textureId);
        cameraTexture.setDefaultBufferSize(width, height);
        cameraTexture.setOnFrameAvailableListener(this::render, null);

        // Get an EGL display connection
        EGLDisplay eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);

        OpenGLUtils.setupEGL(eglDisplay, outgoingSurface);

        // Create initial framebuffers
        for (int i = 0; i < FBO_COUNT; i++) {
            fboRingBuffer[i] = OpenGLUtils.createFBO(width, height);
        }

        // Set up vertex data
        vertexBuffer = OpenGLUtils.setupCoords(squareCoords);
        textureBuffer = OpenGLUtils.setupCoords(textureCoords);

        int glWidth = width / 2;
        int glHeight = height / 2;

        ByteBuffer tb2 = ByteBuffer.allocateDirect(glWidth * glHeight * 2 * 4);
        tb2.order(ByteOrder.nativeOrder());
        textureBufferForFBO = tb2.asFloatBuffer();
        getTextureCoords(textureBufferForFBO, glWidth, glHeight);
        textureBufferForFBO.position(0);

        ByteBuffer tb3 = ByteBuffer.allocateDirect(glWidth * glHeight * 2 * 4);
        tb3.order(ByteOrder.nativeOrder());
        vertexBufferForFBO = tb3.asFloatBuffer();
        vertexBufferForFBO.position(0);

        //Get index buffers
        ByteBuffer ib = ByteBuffer.allocateDirect(glWidth * glHeight * 6 * 4);  // 6 indices per quad, each index is 4 bytes
        ib.order(ByteOrder.nativeOrder());
        indexBufferForFBO = ib.asIntBuffer();
        getTriangleIndexBuffer(indexBufferForFBO, glWidth, glHeight);
        indexBufferForFBO.position(0);

        // Set up shaders and program
        vertexShader = OpenGLUtils.createShader(OpenGLUtils.vertexShaderCode, GLES20.GL_VERTEX_SHADER);
        fragmentShader = OpenGLUtils.createShader(OpenGLUtils.fragmentShaderCode, GLES20.GL_FRAGMENT_SHADER);
        shaderProgram = OpenGLUtils.setupProgram(vertexShader, fragmentShader);
        positionHandle = OpenGLUtils.getHandle(shaderProgram, "vPosition", true);
        textureCoordHandle = OpenGLUtils.getHandle(shaderProgram, "inputTextureCoordinate", true);

        // Set up the second shader program
        fragmentShader2 = OpenGLUtils.createShader(OpenGLUtils.fragmentShaderCode2, GLES20.GL_FRAGMENT_SHADER);
        shaderProgram2 = OpenGLUtils.setupProgram(vertexShader, fragmentShader2);
        positionHandle2 = OpenGLUtils.getHandle(shaderProgram, "vPosition", true);
        textureCoordHandle2 = OpenGLUtils.getHandle(shaderProgram, "inputTextureCoordinate", true);

        // Generate index buffer for OpenGL
        int[] buffers = new int[1];
        GLES20.glGenBuffers(1, buffers, 0);
        indexBufferId = buffers[0];

        // Bind to the buffer. Future commands will affect this buffer specifically.
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, indexBufferId);

        // Transfer data from client memory to the GPU buffer.
        GLES20.glBufferData(GLES20.GL_ELEMENT_ARRAY_BUFFER, glWidth * glHeight * 6 * 4, indexBufferForFBO, GLES20.GL_STATIC_DRAW);

        // Unbind from the buffer when we're done with it.
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);
        isInitialized = true;
    }

    /**
     * @brief Called by host to fetch the SurfaceTexture for receiving camera frames.
     *
     * @return Returns the SurfaceTexture which is used for receiving camera frames.
     */
    public SurfaceTexture getCameraTexture() {
        return cameraTexture;
    }

    /**
     * Sets the outgoing surface along with its dimensions. If OpenGLProcessor is not yet initialized,
     * this method also triggers the initialization.
     *
     * @param outgoingSurface       The surface where the output is to be drawn.
     * @param outgoingSurfaceWidth  The width of the outgoing surface.
     * @param outgoingSurfaceHeight The height of the outgoing surface.
     */
    public void setOutgoingSurface(Surface outgoingSurface, int outgoingSurfaceWidth, int outgoingSurfaceHeight) {
        this.outgoingSurface = outgoingSurface;
        this.outgoingSurfaceWidth = outgoingSurfaceWidth;
        this.outgoingSurfaceHeight = outgoingSurfaceHeight;
        if (!isInitialized) {
            initialize();
        }
    }

    /**
     * Renders the camera frames to the frame buffer object (FBO) and the outgoing surface.
     * This method updates the camera texture, gets the transformation matrix from the SurfaceTexture,
     * then prepares and renders the incoming camera frame to the current FBO.
     * Afterwards, it prepares to render the next FBO to the screen and draws the texture from the next FBO to the screen.
     * It then updates the current FBO index and finally sends the frame to the outgoing surface if available.
     *
     * @param ignored The SurfaceTexture parameter which is not used in this method.
     */
    private void render(SurfaceTexture ignored) {
        renderCameraToFBO();

        long timestamp;
        float[] opticalFlowVerices;
        if (OpticalFlowVertexBuffer.size() == 0) {
            return;
        }
        try {
            OpticalFlowVertices oldestRec = OpticalFlowVertexBuffer.take();
            timestamp = oldestRec.timeStamp;
            opticalFlowVerices = oldestRec.vertices;
        } catch (InterruptedException e) {
            e.printStackTrace();
            return;
        }
        renderFBOToScreen(timestamp, opticalFlowVerices);
    }

    /**
     * @brief This method renders the camera frames to the frame buffer object (FBO).
     *
     * This method updates the camera texture and sets its timestamp. Then it binds to the shader program and sets the viewport.
     * After that, it binds to the framebuffer and the texture, and sets the vertex and texture coordinates. Finally, it draws the texture onto the FBO and then disables the vertex attributes.
     */
    private void renderCameraToFBO() {
        // Update the camera preview texture
        cameraTexture.updateTexImage();
        fboRingBuffer[currentFBOIndex].timestamp = cameraTexture.getTimestamp();

        // Bind to our program
        GLES20.glUseProgram(shaderProgram);
        GLES20.glViewport(0, 0, width, height);

        // Render incoming camera frame to current FBO
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboRingBuffer[currentFBOIndex].framebuffer);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);

        GLES20.glEnableVertexAttribArray(positionHandle);
        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 8, vertexBuffer);
        GLES20.glEnableVertexAttribArray(textureCoordHandle);
        GLES20.glVertexAttribPointer(textureCoordHandle, 2, GLES20.GL_FLOAT, false, 8, textureBuffer);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        GLES20.glDisableVertexAttribArray(positionHandle);
        GLES20.glDisableVertexAttribArray(textureCoordHandle);
    }

    /**
     * @brief This method renders the frames from the frame buffer object (FBO) to the screen.
     *
     * This method prepares the vertex, texture, and index buffers and sets the viewport. Then it finds the FBO with the closest timestamp and prepares to render it to the screen.
     * After that, it binds to the framebuffer and the texture, and sets the vertex and texture coordinates. Finally, it draws the texture from the FBO onto the screen and then disables the vertex attributes.
     *
     * @param timestamp The timestamp of the frame that's being rendered.
     * @param opticalFlowVerices The vertices of the optical flow.
     */
    private void renderFBOToScreen(long timestamp, float[] opticalFlowVerices) {
        vertexBufferForFBO.rewind();
        vertexBufferForFBO.put(opticalFlowVerices);
        vertexBufferForFBO.rewind();
        textureBufferForFBO.rewind();
        indexBufferForFBO.rewind();
        int glWidth = width / 2;
        int glHeight = height / 2;

        // Prepare to render the next FBO to the screen
        long minTimeStampDiff = Math.abs(timestamp);
        int fboIdx = 0;
        for (int i = 0; i < FBO_COUNT; i++) {
            long temp = Math.abs(timestamp - fboRingBuffer[i].timestamp);
            if (temp < minTimeStampDiff) {
                minTimeStampDiff = temp;
                fboIdx = i;
            }
        }

        int nextFBOIndex = fboIdx;

        GLES20.glUseProgram(shaderProgram2);
        GLES20.glViewport(0, 0, outgoingSurfaceWidth, outgoingSurfaceHeight);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboRingBuffer[nextFBOIndex].texture);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, indexBufferId);
        GLES20.glEnableVertexAttribArray(positionHandle2);
        GLES20.glVertexAttribPointer(positionHandle2, 2, GLES20.GL_FLOAT, false, 8, vertexBufferForFBO);
        GLES20.glEnableVertexAttribArray(textureCoordHandle2);
        GLES20.glVertexAttribPointer(textureCoordHandle2, 2, GLES20.GL_FLOAT, false, 8, textureBufferForFBO);

        GLES20.glDrawElements(GLES20.GL_TRIANGLES, glWidth * glHeight * 6, GLES20.GL_UNSIGNED_INT, 0);

        GLES20.glDisableVertexAttribArray(positionHandle2);
        GLES20.glDisableVertexAttribArray(textureCoordHandle2);
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);

        // Update the current FBO index
        currentFBOIndex = (currentFBOIndex + 1) % FBO_COUNT;

        // Send the frame to the outgoing Surface
        if (outgoingSurface != null) {
            EGL14.eglSwapBuffers(EGL14.eglGetCurrentDisplay(), EGL14.eglGetCurrentSurface(EGL14.EGL_DRAW));
        }
    }

    /**
     * @brief This method adds a new set of optical flow vertices to the queue.
     *
     * This method creates a new OpticalFlowVertices object and adds it to the queue. If the size of the queue is greater than 8, it removes the oldest item from the queue.
     *
     * @param timestamp The timestamp of the frame.
     * @param vertices The vertices of the optical flow.
     */
    public void addOpticalFlowVertices(long timestamp, float[] vertices) {
        OpticalFlowVertices rec = new OpticalFlowVertices(timestamp, vertices);
        OpticalFlowVertexBuffer.add(rec);
        if (OpticalFlowVertexBuffer.size() > 8) {
            OpticalFlowVertexBuffer.poll();
        }
    }

    /**
     * @class OpticalFlowVertices
     * @brief This class is used for storing the vertices of the optical flow and their corresponding timestamp.
     *
     * The class implements Comparable, so that the instances of this class can be sorted based on their timestamp.
     */
    private class OpticalFlowVertices implements Comparable<OpticalFlowVertices> {
        public long timeStamp;
        public float[] vertices;

        public OpticalFlowVertices(long inTimestamp, float[] inVertices) {
            timeStamp = inTimestamp;
            vertices = inVertices;
        }

        @Override
        public int compareTo(OpticalFlowVertices other) {
            return Long.compare(this.timeStamp, other.timeStamp);
        }
    }

}
