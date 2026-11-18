#version 450

layout(location = 0) in vec3 in_position;

layout(std140, set = 0, binding = 0) uniform ShadowUniforms {
    mat4 shadow_projection;
};

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
};

void main() {
    gl_Position = shadow_projection * model * vec4(in_position, 1.0);
}