package com.example.dualphoneshooter;

import android.graphics.Path;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.EGL14;

import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.util.Log;
import android.view.Surface;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.concurrent.PriorityBlockingQueue;

/**
 * @class OpenGLProcessor
 * @brief This class handles OpenGL related operations for DualPhoneShooter.
 */
public class OpenGLProcessor {
    // Static configuration variables
    private static final String TAG = "OpenGLProcessor";
    private static final int FBO_COUNT = 8;
    private final int width;
    private final int height;
    // Vertex shader
    private final String vertexShaderCode =
            "attribute vec4 vPosition;" +
                    "attribute vec2 inputTextureCoordinate;" +
                    "varying vec2 textureCoordinate;" +
                    "void main() {" +
                    "  textureCoordinate = inputTextureCoordinate;" +
                    "  gl_Position = vPosition;" +
                    "}";
    // Fragment shader code for camera preview texture
    private final String fragmentShaderCode =
            "#extension GL_OES_EGL_image_external : require\n" +
                    "precision mediump float;" +
                    "varying vec2 textureCoordinate;\n" +
                    "uniform samplerExternalOES s_texture;\n" +
                    "void main() {" +
                    "  gl_FragColor = texture2D( s_texture, textureCoordinate );\n" +
                    "}";
    // Fragment shader for second rendering pass
    private final String fragmentShaderCode2 =
            "precision mediump float;" +
                    "varying vec2 textureCoordinate;\n" +
                    "uniform sampler2D s_texture;\n" +
                    "void main() {" +
                    "  gl_FragColor = texture2D( s_texture, textureCoordinate );\n" +
                    "}";
    private final Framebuffer[] fboRingBuffer = new Framebuffer[FBO_COUNT];
    // A queue that's sort by timestamp.
    PriorityBlockingQueue<OpticalFlowVertices> OpticalFlowVertexBuffer = new PriorityBlockingQueue<>();
    private EGLContext eglContext;
    // Rectangle vertices
    private float squareCoords[] = {
            -1.0f, 1.0f,   // top left
            -1.0f, -1.0f,  // bottom left
            1.0f, 1.0f,    // top right
            1.0f, -1.0f}; // bottom right
    // Texture coordinates
    // Note: In OpenGL, texture coordinates are specified with (0, 0) at the bottom-left corner,
    // and (1, 1) at the top-right. However, the SurfaceTexture from the camera has (0, 0)
    // at the top-left and (1, 1) at the bottom-right, which causes the upside-down image.
    // Texture coordinates
    private float textureCoords[] = {
            0.0f, 0.0f,  // top left
            0.0f, 1.0f,  // bottom left
            1.0f, 0.0f,  // top right
            1.0f, 1.0f}; // bottom right
    // Vertex buffers, Texture buffers, and Index buffers
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
    private int textureHandle;
    private int textureCoordHandle;
    private int fragmentShader2;
    private int shaderProgram2;
    private int positionHandle2;
    private int textureHandle2;
    private int textureCoordHandle2;
    private boolean isInitialized;

    /**
     * @param width  Width of the OpenGL context.
     * @param height Height of the OpenGL context.
     * @brief Constructor for OpenGLProcessor.
     */
    public OpenGLProcessor(int width, int height) {
        // Initialize dimensions
        this.width = width;
        this.height = height;
        isInitialized = false;
    }

    private native void getTextureCoords(FloatBuffer buffer, int w, int h);

    private native void getTriangleIndexBuffer(IntBuffer buffer, int w, int h);

    /**
     * @brief Init code. Called only after outgoingSurface is available.
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
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            throw new RuntimeException("EGL_NO_DISPLAY");
        }

        // Initialize the EGL display connection
        int[] version = new int[2];
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            throw new RuntimeException("eglInitialize failed");
        }

        // Configure the EGL framebuffer and rendering surfaces
        int[] attribList = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_NONE
        };

        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        if (!EGL14.eglChooseConfig(eglDisplay, attribList, 0, configs, 0, configs.length, numConfigs, 0)) {
            throw new RuntimeException("eglChooseConfig failed");
        }

        // Create an EGL rendering context
        int[] attrib_list = {
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
                EGL14.EGL_NONE
        };

        EGLContext eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, attrib_list, 0);
        if (eglContext == null) {
            throw new RuntimeException("eglCreateContext failed");
        }

        // Create a surface for the context
        EGLSurface eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, configs[0], outgoingSurface, new int[]{EGL14.EGL_NONE}, 0);
        if (eglSurface == null) {
            throw new RuntimeException("eglCreateWindowSurface failed");
        }

        // Make the context current
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            throw new RuntimeException("eglMakeCurrent failed");
        }

        // Create initial framebuffers
        for (int i = 0; i < FBO_COUNT; i++) {
            fboRingBuffer[i] = createFBO();
        }

        // Set up vertex data
        ByteBuffer bb = ByteBuffer.allocateDirect(squareCoords.length * 4);
        bb.order(ByteOrder.nativeOrder());
        vertexBuffer = bb.asFloatBuffer();
        vertexBuffer.put(squareCoords);
        vertexBuffer.position(0);

        ByteBuffer tb = ByteBuffer.allocateDirect(textureCoords.length * 4);
        tb.order(ByteOrder.nativeOrder());
        textureBuffer = tb.asFloatBuffer();
        textureBuffer.put(textureCoords);
        textureBuffer.position(0);

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
        vertexShader = GLES20.glCreateShader(GLES20.GL_VERTEX_SHADER);
        GLES20.glShaderSource(vertexShader, vertexShaderCode);
        GLES20.glCompileShader(vertexShader);

        fragmentShader = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER);
        GLES20.glShaderSource(fragmentShader, fragmentShaderCode);
        GLES20.glCompileShader(fragmentShader);

        shaderProgram = GLES20.glCreateProgram();
        GLES20.glAttachShader(shaderProgram, vertexShader);
        GLES20.glAttachShader(shaderProgram, fragmentShader);
        GLES20.glLinkProgram(shaderProgram);

        // Get handles for passing in data to the shaders
        positionHandle = GLES20.glGetAttribLocation(shaderProgram, "vPosition");
        textureHandle = GLES20.glGetUniformLocation(shaderProgram, "s_texture");
        textureCoordHandle = GLES20.glGetAttribLocation(shaderProgram, "inputTextureCoordinate");

        // Set up the second shader program
        fragmentShader2 = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER);
        GLES20.glShaderSource(fragmentShader2, fragmentShaderCode2);
        GLES20.glCompileShader(fragmentShader2);

        shaderProgram2 = GLES20.glCreateProgram();
        GLES20.glAttachShader(shaderProgram2, vertexShader);
        GLES20.glAttachShader(shaderProgram2, fragmentShader2);
        GLES20.glLinkProgram(shaderProgram2);

        // Get handle for passing in data to the second shader
        positionHandle2 = GLES20.glGetAttribLocation(shaderProgram2, "vPosition");
        textureHandle2 = GLES20.glGetUniformLocation(shaderProgram2, "s_texture");
        textureCoordHandle2 = GLES20.glGetAttribLocation(shaderProgram2, "inputTextureCoordinate");

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
     * @brief Method to create a FBO with underlying texture
     */
    private Framebuffer createFBO() {
        int[] framebuffers = new int[1];
        GLES20.glGenFramebuffers(1, framebuffers, 0);

        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textures[0]);

        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);

        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, framebuffers[0]);

        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, textures[0], 0);

        int status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
        if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "Failed to create framebuffer, status: " + status);
            return null;
        }

        return new Framebuffer(framebuffers[0], textures[0]);
    }

    /**
     * @brief Called by host to fetch the SurfaceTexture for receiving camera frames.
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
        // Update the camera preview texture
        cameraTexture.updateTexImage();
        fboRingBuffer[currentFBOIndex].timestamp = cameraTexture.getTimestamp();

        // Get transformation matrix from the SurfaceTexture
        float[] mtx = new float[16];
        cameraTexture.getTransformMatrix(mtx);
        // Bind to our program
        GLES20.glUseProgram(shaderProgram);
        GLES20.glViewport(0, 0, width, height);
        // Render incoming camera frame to current FBO
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboRingBuffer[currentFBOIndex].framebuffer);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glUniform1i(textureHandle, 0);
        GLES20.glEnableVertexAttribArray(positionHandle);
        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 8, vertexBuffer);
        GLES20.glEnableVertexAttribArray(textureCoordHandle);
        GLES20.glVertexAttribPointer(textureCoordHandle, 2, GLES20.GL_FLOAT, false, 8, textureBuffer);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glDisableVertexAttribArray(positionHandle);
        GLES20.glDisableVertexAttribArray(textureCoordHandle);


        //Get ...
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
        vertexBufferForFBO.rewind();
        vertexBufferForFBO.put(opticalFlowVerices);
        vertexBufferForFBO.rewind();
        textureBufferForFBO.rewind();
        indexBufferForFBO.rewind();
        int glWidth = width / 2;
        int glHeight = height / 2;
        GLES20.glUseProgram(shaderProgram2);
        GLES20.glViewport(0, 0, outgoingSurfaceWidth, outgoingSurfaceHeight);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboRingBuffer[nextFBOIndex].texture);
        GLES20.glUniform1i(textureHandle2, 0);

        // Draw the texture from the next FBO to the screen
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, indexBufferId);
        GLES20.glEnableVertexAttribArray(positionHandle2);
        GLES20.glVertexAttribPointer(positionHandle2, 2, GLES20.GL_FLOAT, false, 8, vertexBufferForFBO);
        GLES20.glEnableVertexAttribArray(textureCoordHandle2);
        GLES20.glVertexAttribPointer(textureCoordHandle2, 2, GLES20.GL_FLOAT, false, 8, textureBufferForFBO); // use textureBufferForFBO here
        // Note: The last argument to glDrawElements is the number of indices, not vertices.
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

    public void addOpticalFlowVertices(long timestamp, float[] vertices) {
        OpticalFlowVertices rec = new OpticalFlowVertices(timestamp, vertices);
        OpticalFlowVertexBuffer.add(rec);
        if (OpticalFlowVertexBuffer.size() > 8) {
            OpticalFlowVertexBuffer.poll();
        }
    }

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

    /**
     * Represents a framebuffer object (FBO) that contains a framebuffer and a texture.
     * The framebuffer is used for rendering purposes, and the texture holds the rendered data.
     */
    private class Framebuffer {
        int framebuffer;
        /**
         * < The ID of the framebuffer object.
         */
        int texture;  /**< The ID of the texture object. */

        /**
         * Constructs a new Framebuffer object with the specified framebuffer and texture IDs.
         *
         * @param framebuffer The ID of the framebuffer object.
         * @param texture     The ID of the texture object.
         */
        long timestamp;

        Framebuffer(int framebuffer, int texture) {
            this.framebuffer = framebuffer;
            this.texture = texture;
            this.timestamp = 0;
        }
    }

}
