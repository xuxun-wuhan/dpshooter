package com.example.dualphoneshooter;

import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.EGL14;

import android.opengl.GLES20;
import android.util.Log;
import android.view.Surface;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * @class OpenGLUtils
 * @brief This class provides utility methods for OpenGL related operations.
 */
public class OpenGLUtils {
    // Static configuration variables
    private static final String TAG = "OpenGLUtils";

    // Vertex shader code
    public static String vertexShaderCode =
            "attribute vec4 vPosition;" +
                    "attribute vec2 inputTextureCoordinate;" +
                    "varying vec2 textureCoordinate;" +
                    "void main() {" +
                    "  textureCoordinate = inputTextureCoordinate;" +
                    "  gl_Position = vPosition;" +
                    "}";

    // Fragment shader code for camera preview texture
    public static String fragmentShaderCode =
            "#extension GL_OES_EGL_image_external : require\n" +
                    "precision mediump float;" +
                    "varying vec2 textureCoordinate;\n" +
                    "uniform samplerExternalOES s_texture;\n" +
                    "void main() {" +
                    "  gl_FragColor = texture2D( s_texture, textureCoordinate );\n" +
                    "}";

    // Fragment shader for second rendering pass
    public static String fragmentShaderCode2 =
            "precision mediump float;" +
                    "varying vec2 textureCoordinate;\n" +
                    "uniform sampler2D s_texture;\n" +
                    "void main() {" +
                    "  gl_FragColor = texture2D( s_texture, textureCoordinate );\n" +
                    "}";

    /**
     * @brief Creates a shader with the given shader code and type.
     * @param shaderCode The source code of the shader.
     * @param shaderType The type of the shader. It can be either GLES20.GL_VERTEX_SHADER or GLES20.GL_FRAGMENT_SHADER.
     * @return The ID of the shader.
     */
    public static int createShader(String shaderCode, int shaderType) {
        int shader = GLES20.glCreateShader(shaderType);
        GLES20.glShaderSource(shader, shaderCode);
        GLES20.glCompileShader(shader);
        return shader;
    }

    /**
     * @brief Creates a Framebuffer Object (FBO) with the given width and height.
     * @param width The width of the FBO.
     * @param height The height of the FBO.
     * @return A Framebuffer object that contains the FBO and the texture.
     */
    public static Framebuffer createFBO(int width, int height) {
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
     * @brief Sets up the EGL display and context.
     * @param eglDisplay The EGL display.
     * @param outgoingSurface The surface where the output is to be drawn.
     */
    public static void setupEGL(EGLDisplay eglDisplay, Surface outgoingSurface) {
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
    }

    /**
     * @brief Sets up a FloatBuffer for the given coordinates.
     * @param coords The coordinates.
     * @return A FloatBuffer that contains the coordinates.
     */
    public static FloatBuffer setupCoords(float[] coords) {
        ByteBuffer bb = ByteBuffer.allocateDirect(coords.length * 4);
        bb.order(ByteOrder.nativeOrder());
        FloatBuffer buffer = bb.asFloatBuffer();
        buffer.put(coords);
        buffer.position(0);
        return buffer;
    }

    /**
     * @brief Sets up an OpenGL program with the given vertex and fragment shaders.
     * @param vertexShader The ID of the vertex shader.
     * @param fragmentShader The ID of the fragment shader.
     * @return The ID of the program.
     */
    public static int setupProgram(int vertexShader, int fragmentShader) {
        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertexShader);
        GLES20.glAttachShader(program, fragmentShader);
        GLES20.glLinkProgram(program);
        return program;
    }

    /**
     * @brief Gets the handle of a shader variable.
     * @param program The ID of the program where the variable is located.
     * @param name The name of the variable.
     * @param isAttribute True if the variable is an attribute, false if it's a uniform.
     * @return The handle of the variable.
     */
    public static int getHandle(int program, String name, boolean isAttribute) {
        return isAttribute ? GLES20.glGetAttribLocation(program, name) : GLES20.glGetUniformLocation(program, name);
    }

    /**
     * @class Framebuffer
     * @brief Represents a framebuffer object (FBO) that contains a framebuffer and a texture.
     *
     * The framebuffer is used for rendering purposes, and the texture holds the rendered data.
     */
    public static class Framebuffer {
        int framebuffer; /**< The ID of the framebuffer object. */
        int texture; /**< The ID of the texture object. */
        long timestamp; /**< The timestamp of the frame. */

        /**
         * @brief Constructs a new Framebuffer object with the specified framebuffer and texture IDs.
         * @param framebuffer The ID of the framebuffer object.
         * @param texture The ID of the texture object.
         */
        Framebuffer(int framebuffer, int texture) {
            this.framebuffer = framebuffer;
            this.texture = texture;
            this.timestamp = 0;
        }
    }
}
