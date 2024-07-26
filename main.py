import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
import glm
import ctypes
import time

vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
out vec3 TexCoord3D;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float slice;
void main() {
    vec3 pos = aPos;
    pos.z = slice * 2.0 - 1.0; // Map slice [0, 1] to [-1, 1] in NDC
    gl_Position = projection * view * model * vec4(pos, 1.0);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y); // Flip Y coordinate
    TexCoord3D = vec3(aTexCoord.x, 1.0 - aTexCoord.y, slice); // Flip Y coordinate
}
"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
in vec3 TexCoord3D;
uniform sampler3D texture1;
void main() {
    float intensity = texture(texture1, TexCoord3D).r;
    FragColor = vec4(intensity, intensity, intensity, intensity * 0.4); // Adjust alpha for better visibility
    if (FragColor.a < 0.01) discard;
}
"""

window_width = 1280
window_height = 720

slices_width = 256
slices_height = 256
num_slices = 128


def load_data_from_file(file_name):
    with open(file_name, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32)
    return data


def load_3d_texture(data, width, height, depth):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_3D, texture_id)
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, width, height, depth, 0, GL_RED, GL_FLOAT, data)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_3D, 0)
    return texture_id


def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        info_log = glGetShaderInfoLog(shader).decode()
        print(f"Shader compilation failed:\n{info_log}")
    return shader


def create_shader_program():
    vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source)
    fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source)
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        info_log = glGetProgramInfoLog(shader_program).decode()
        print(f"Shader program linking failed:\n{info_log}")
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return shader_program


def render_slices(texture, shader_program, vao, model, view, projection, num_slices):
    glUseProgram(shader_program)

    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    projection_loc = glGetUniformLocation(shader_program, "projection")
    slice_loc = glGetUniformLocation(shader_program, "slice")

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))

    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glBindTexture(GL_TEXTURE_3D, texture)
    glBindVertexArray(vao)

    for i in range(num_slices - 1, -1, -1):
        slice_val = i / (num_slices - 1)
        glUniform1f(slice_loc, slice_val)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    glDisable(GL_BLEND)
    glDepthMask(GL_TRUE)
    glDisable(GL_DEPTH_TEST)
    glBindVertexArray(0)


def mouse_button_callback(window, button, action, mods):
    global mouse_pressed, first_mouse
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            mouse_pressed = True
        elif action == glfw.RELEASE:
            mouse_pressed = False
            first_mouse = True


def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)


def mouse_callback(window, xpos, ypos):
    global last_mouse_pos, first_mouse, rotation

    if not mouse_pressed:
        return

    current_mouse_pos = glm.vec2(xpos, ypos)

    if first_mouse:
        last_mouse_pos = current_mouse_pos
        first_mouse = False
        return

    mouse_delta = current_mouse_pos - last_mouse_pos
    last_mouse_pos = current_mouse_pos

    rotation_speed = 0.005
    yaw_quat = glm.angleAxis(mouse_delta.x * rotation_speed, glm.vec3(0.0, 1.0, 0.0))
    pitch_quat = glm.angleAxis(mouse_delta.y * rotation_speed, glm.vec3(1.0, 0.0, 0.0))

    rotation = yaw_quat * pitch_quat * rotation
    rotation = glm.normalize(rotation)


def scroll_callback(window, xoffset, yoffset):
    global zoom
    zoom_sensitivity = 0.1
    zoom -= yoffset * zoom_sensitivity
    zoom = glm.clamp(zoom, 0.5, 5.0)


def main():
    global mouse_pressed, first_mouse, last_mouse_pos, rotation, zoom
    mouse_pressed = False
    first_mouse = True
    last_mouse_pos = glm.vec2(slices_width / 2.0, slices_height / 2.0)
    rotation = glm.quat(1.0, 0.0, 0.0, 0.0)
    zoom = 3.75

    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    window = glfw.create_window(window_width, window_height, "3D Volume Viewer Prototype", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # Generate random floating-point values for the 3D texture
    with open("./engine_256x256x128_uint8.raw", 'rb') as file:
        binary_data = file.read()
        file.close()

    data = np.frombuffer(binary_data, dtype=np.uint8)

    # Load data and texture
    texture_id = load_3d_texture(data, slices_width, slices_height, num_slices)
    shader_program = create_shader_program()

    vertices = np.array([
        1.0, 1.0, 0.0, 1.0, 1.0,
        1.0, -1.0, 0.0, 1.0, 0.0,
        -1.0, 1.0, 0.0, 0.0, 1.0,
        -1.0, -1.0, 0.0, 0.0, 0.0
    ], dtype=np.float32)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float),
                          ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    glEnable(GL_DEPTH_TEST)

    target_fps = 60.0
    target_frame_time = 1.0 / target_fps
    last_frame_time = time.time()

    while not glfw.window_should_close(window):
        current_frame_time = time.time()
        delta_time = current_frame_time - last_frame_time

        if delta_time >= target_frame_time:
            last_frame_time = current_frame_time

            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            view = glm.lookAt(glm.vec3(0.0, 0.0, zoom), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
            projection = glm.perspective(glm.radians(45.0), window_width / window_height, 0.1, 100.0)
            model = glm.mat4_cast(rotation)

            render_slices(texture_id, shader_program, vao, model, view, projection, num_slices)

            glfw.swap_buffers(window)
            glfw.poll_events()
        else:
            time.sleep(0.001)

    glDeleteVertexArrays(1, vao)
    glDeleteBuffers(1, vbo)
    glDeleteProgram(shader_program)
    glDeleteTextures(1, texture_id)

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
