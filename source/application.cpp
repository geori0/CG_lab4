#include "application.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <lodepng.h>

namespace application {
namespace {

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct SceneUniforms {
    alignas(16) glm::mat4 view_projection;
    alignas(16) glm::mat4 dir_shadow_projection;
    alignas(16) glm::mat4 spot_shadow_projection;
    alignas(16) glm::vec3 camera_position;
    alignas(16) glm::vec3 directional_light_direction;
    alignas(16) glm::vec3 directional_light_color;
    alignas(16) glm::vec3 ambient_light;
    uint32_t point_light_count;
    uint32_t spot_light_count;
    int32_t shadow_spot_index;
    uint32_t dir_shadow_enabled;
};

struct GpuPointLight {
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 color;
};

struct GpuSpotLight {
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 color;
    alignas(16) glm::vec3 direction;
    float inner_cos;
    float outer_cos;
};

struct ModelUniforms {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 normal_matrix;
    alignas(16) glm::vec3 albedo;
    alignas(16) glm::vec3 specular;
    float shininess;
    uint32_t unlit;
};

struct PushConstants {
    uint32_t effect;
    float time;
    float strength;
    float padding;
};

static_assert(offsetof(SceneUniforms, dir_shadow_projection) == 64);
static_assert(offsetof(SceneUniforms, spot_shadow_projection) == 128);
static_assert(offsetof(SceneUniforms, camera_position) == 192);
static_assert(offsetof(SceneUniforms, directional_light_direction) == 208);
static_assert(offsetof(SceneUniforms, directional_light_color) == 224);
static_assert(offsetof(SceneUniforms, ambient_light) == 240);
static_assert(offsetof(SceneUniforms, point_light_count) == 252);
static_assert(offsetof(SceneUniforms, spot_light_count) == 256);
static_assert(offsetof(SceneUniforms, shadow_spot_index) == 260);
static_assert(offsetof(SceneUniforms, dir_shadow_enabled) == 264);

static_assert(sizeof(GpuPointLight) == 32);
static_assert(offsetof(GpuPointLight, color) == 16);

static_assert(sizeof(GpuSpotLight) == 64);
static_assert(offsetof(GpuSpotLight, direction) == 32);
static_assert(offsetof(GpuSpotLight, inner_cos) == 44);
static_assert(offsetof(GpuSpotLight, outer_cos) == 48);

static_assert(offsetof(ModelUniforms, normal_matrix) == 64);
static_assert(offsetof(ModelUniforms, albedo) == 128);
static_assert(offsetof(ModelUniforms, specular) == 144);
static_assert(offsetof(ModelUniforms, shininess) == 156);
static_assert(offsetof(ModelUniforms, unlit) == 160);

static_assert(sizeof(PushConstants) == 16);

enum SamplingEffect {
    EFFECT_NONE = 0,
    EFFECT_WOBBLE,
    EFFECT_SWIRL,
    EFFECT_WIDE_ANGLE,
    EFFECT_KALEIDOSCOPE,
    EFFECT_FROSTED_GLASS,
    EFFECT_ANAGLYPH,
};

const char* const EFFECT_NAMES[] = {
    "None", "Wobble", "Swirl", "Wide angle", "Kaleidoscope", "Frosted glass", "Anaglyph (red/cyan)",
};

struct MaterialDefinition {
    const char* name;
    const char* albedo;
    const char* specular;
    const char* emissive;
};

constexpr MaterialDefinition MATERIAL_DEFINITIONS[] = {
    { "Planet",      "planet_albedo.png",  "planet_specular.png",  "planet_emissive.png" },
    { "Lava",        "lava_albedo.png",    "lava_specular.png",    "lava_emissive.png" },
    { "Bricks",      "bricks_albedo.png",  "bricks_specular.png",  nullptr },
    { "Stone floor", "floor_albedo.png",   "floor_specular.png",   nullptr },
    { "Checker",     "checker_albedo.png", nullptr,                nullptr },
    { "Lenna",       "lenna.png",          nullptr,                nullptr },
};
constexpr uint32_t MATERIAL_COUNT = sizeof(MATERIAL_DEFINITIONS) / sizeof(MATERIAL_DEFINITIONS[0]);

struct ObjectSettings {
    int material;
    int effect;
    float effect_strength;
    glm::vec3 albedo_tint;
    glm::vec3 specular_tint;
    float shininess;
};

struct Texture {
    VkImage image;
    VmaAllocation allocation;
    VkImageView view;
};

struct ShadowMap {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkBuffer ubo_buffer = VK_NULL_HANDLE;
    VmaAllocation ubo_alloc = VK_NULL_HANDLE;
    void* ubo_mapped = nullptr;
    VkDescriptorSet scene_set = VK_NULL_HANDLE;
};

struct PointLightSettings {
    glm::vec3 position;
    glm::vec3 color;
    float intensity;
};

struct SpotLightSettings {
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 color;
    float intensity;
    float inner_angle;
    float outer_angle;
};

constexpr uint32_t MAX_POINT_LIGHTS = 16;
constexpr uint32_t MAX_SPOT_LIGHTS = 16;

constexpr uint32_t OBJECT_SPHERE_1 = 0;
constexpr uint32_t OBJECT_SPHERE_2 = 1;
constexpr uint32_t OBJECT_FLOOR = 2;
constexpr uint32_t OBJECT_WALL = 3;
constexpr uint32_t SCENE_OBJECT_COUNT = 4;
constexpr uint32_t POINT_MARKER_BASE = SCENE_OBJECT_COUNT;
constexpr uint32_t SPOT_MARKER_BASE = POINT_MARKER_BASE + MAX_POINT_LIGHTS;
constexpr uint32_t SPOT_CONE_BASE = SPOT_MARKER_BASE + MAX_SPOT_LIGHTS;
constexpr uint32_t MODEL_COUNT = SPOT_CONE_BASE + MAX_SPOT_LIGHTS;

constexpr float LIGHT_MARKER_SCALE = 0.12f;
constexpr float SPOT_CONE_LENGTH = 2.0f;
constexpr float FLOOR_HALF_SIZE = 8.0f;
constexpr float FLOOR_UV_REPEAT = 4.0f;
constexpr uint32_t SHADOW_MAP_SIZE = 4096;

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_buffer_alloc = VK_NULL_HANDLE;
VkBuffer index_buffer = VK_NULL_HANDLE;
VmaAllocation index_buffer_alloc = VK_NULL_HANDLE;
uint32_t sphere_index_count = 0;
uint32_t floor_first_index = 0;
constexpr uint32_t FLOOR_INDEX_COUNT = 6;

VkBuffer cone_vertex_buffer = VK_NULL_HANDLE;
VmaAllocation cone_vertex_buffer_alloc = VK_NULL_HANDLE;
uint32_t cone_vertex_count = 0;

VkBuffer scene_ubo_buffer = VK_NULL_HANDLE;
VmaAllocation scene_ubo_alloc = VK_NULL_HANDLE;
void* scene_ubo_mapped = nullptr;

VkBuffer point_lights_buffer = VK_NULL_HANDLE;
VmaAllocation point_lights_alloc = VK_NULL_HANDLE;
void* point_lights_mapped = nullptr;

VkBuffer spot_lights_buffer = VK_NULL_HANDLE;
VmaAllocation spot_lights_alloc = VK_NULL_HANDLE;
void* spot_lights_mapped = nullptr;

VkBuffer model_ubo_buffer = VK_NULL_HANDLE;
VmaAllocation model_ubo_alloc = VK_NULL_HANDLE;
uint8_t* model_ubo_mapped = nullptr;
VkDeviceSize model_ubo_stride = 0;

VkSampler texture_sampler = VK_NULL_HANDLE;
VkSampler shadow_sampler = VK_NULL_HANDLE;
VkFormat shadow_depth_format = VK_FORMAT_D32_SFLOAT;
ShadowMap dir_shadow;
ShadowMap spot_shadow;

std::vector<Texture> textures;
std::vector<const char*> material_names;

VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout material_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;
VkDescriptorSet material_sets[MATERIAL_COUNT] = {};

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline triangle_pipeline = VK_NULL_HANDLE;
VkPipeline line_pipeline = VK_NULL_HANDLE;
VkPipeline shadow_pipeline = VK_NULL_HANDLE;

bool use_perspective = true;
float fov = 45.0f;
float ortho_size = 3.5f;
float near_plane = 0.1f;
float far_plane = 100.0f;
glm::vec3 camera_pos = glm::vec3(0.0f, 2.5f, 6.0f);
float camera_yaw = 0.0f;
float camera_pitch = -23.0f;

glm::vec3 ambient_light = glm::vec3(0.04f, 0.04f, 0.06f);

bool dir_light_enabled = true;
glm::vec3 dir_light_direction = glm::vec3(-0.4f, -1.0f, -0.3f);
glm::vec3 dir_light_color = glm::vec3(1.0f, 1.0f, 1.0f);
float dir_light_intensity = 0.55f;

bool dir_shadow_enabled = true;
float shadow_ortho_size = 8.0f;
float shadow_near_plane = 0.1f;
float shadow_far_plane = 40.0f;
float shadow_light_distance = 15.0f;
bool shadow_follow_camera = false;
float depth_bias_constant = 1.25f;
float depth_bias_slope = 1.75f;

bool spot_shadow_enabled = true;
int shadow_spot_index = 0;
float spot_shadow_near = 0.2f;
float spot_shadow_far = 25.0f;

std::vector<PointLightSettings> point_lights = {
    { glm::vec3(2.5f, 1.5f, 2.0f), glm::vec3(1.0f, 0.75f, 0.45f), 4.0f },
};

std::vector<SpotLightSettings> spot_lights = {
    { glm::vec3(-2.5f, 2.5f, 1.0f), glm::vec3(2.5f, -2.5f, -1.0f),
      glm::vec3(1.0f, 1.0f, 1.0f), 14.0f, 15.0f, 28.0f },
};

bool show_lights = true;

ObjectSettings objects[SCENE_OBJECT_COUNT] = {
    { 0, EFFECT_NONE,   1.0f, glm::vec3(1.0f), glm::vec3(1.0f), 32.0f },
    { 1, EFFECT_WOBBLE, 1.0f, glm::vec3(1.0f), glm::vec3(1.0f), 64.0f },
    { 3, EFFECT_NONE,   1.0f, glm::vec3(1.0f), glm::vec3(1.0f), 16.0f },
    { 2, EFFECT_NONE,   1.0f, glm::vec3(1.0f), glm::vec3(1.0f), 24.0f },
};

glm::vec3 obj1_pos = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 obj1_rot = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 obj1_scale = glm::vec3(1.0f, 1.0f, 1.0f);

bool anim_playing = true;
float anim_speed = 1.0f;
double anim_time = 0.0;
double last_time = 0.0;
bool use_trajectory = false;
float traj_radius = 2.0f;

bool obj2_enabled = true;
glm::vec3 obj2_scale = glm::vec3(0.5f, 0.5f, 0.5f);

bool floor_enabled = true;
float floor_y = -1.5f;

bool wall_enabled = true;
float wall_z = -5.0f;

std::vector<char> readShaderFile(const std::string& filename, bool quiet = false) {
    std::vector<std::string> search_paths = { filename, "shaders/" + filename, "../shaders/" + filename };
    for (const auto& path : search_paths) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (file.is_open()) {
            size_t fileSize = (size_t)file.tellg();
            std::vector<char> buffer(fileSize);
            file.seekg(0);
            file.read(buffer.data(), fileSize);
            file.close();
            return buffer;
        }
    }
    if (!quiet) {
        std::cerr << "Failed to find shader: " << filename << '\n';
    }
    return {};
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data()),
    };
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
    return shaderModule;
}

glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback) {
    float length = glm::length(v);
    return length > 1e-6f ? v / length : fallback;
}

glm::vec3 chooseUpVector(const glm::vec3& dir) {
    return std::abs(dir.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(0.0f, 0.0f, 1.0f);
}

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

VkPhysicalDevice getPhysicalDevice() {
    VmaAllocatorInfo allocator_info = {};
    vmaGetAllocatorInfo(graphics::internal::context.allocator, &allocator_info);
    return allocator_info.physicalDevice;
}

uint32_t findGraphicsQueueFamily(VkPhysicalDevice physical_device) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());
    for (uint32_t i = 0; i < count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) return i;
    }
    return 0;
}

VkFormat findDepthFormat(VkPhysicalDevice physical_device) {
    const VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
    };
    const VkFormatFeatureFlags required_features =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;

    for (VkFormat format : candidates) {
        VkFormatProperties props = {};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);
        if ((props.optimalTilingFeatures & required_features) == required_features) {
            return format;
        }
    }
    return VK_FORMAT_D32_SFLOAT;
}

VkImageAspectFlags getDepthBarrierAspectMask(VkFormat format) {
    if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    return VK_IMAGE_ASPECT_DEPTH_BIT;
}

void generateSphere(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
    const uint32_t stacks = 20;
    const uint32_t slices = 24;
    const float radius = 1.0f;

    for (uint32_t i = 0; i <= stacks; ++i) {
        float phi = glm::pi<float>() * float(i) / float(stacks);
        float y = radius * std::cos(phi);
        float r_sin_phi = radius * std::sin(phi);

        for (uint32_t j = 0; j <= slices; ++j) {
            float theta = 2.0f * glm::pi<float>() * float(j) / float(slices);
            glm::vec3 pos(r_sin_phi * std::cos(theta), y, r_sin_phi * std::sin(theta));
            glm::vec2 uv(1.0f - float(j) / float(slices), float(i) / float(stacks));
            vertices.push_back({ pos, glm::normalize(pos), uv });
        }
    }

    for (uint32_t i = 0; i < stacks; ++i) {
        for (uint32_t j = 0; j < slices; ++j) {
            uint16_t a = static_cast<uint16_t>(i * (slices + 1) + j);
            uint16_t b = a + 1;
            uint16_t c = static_cast<uint16_t>(a + slices + 1);
            uint16_t d = c + 1;

            if (i != stacks - 1) {
                indices.push_back(a);
                indices.push_back(d);
                indices.push_back(c);
            }
            if (i != 0) {
                indices.push_back(a);
                indices.push_back(b);
                indices.push_back(d);
            }
        }
    }
}

void generateFloor(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
    const float h = FLOOR_HALF_SIZE;
    const float t = FLOOR_UV_REPEAT;
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    uint16_t base = static_cast<uint16_t>(vertices.size());

    vertices.push_back({ glm::vec3(-h, 0.0f, -h), up, glm::vec2(0.0f, 0.0f) });
    vertices.push_back({ glm::vec3(-h, 0.0f,  h), up, glm::vec2(0.0f, t) });
    vertices.push_back({ glm::vec3( h, 0.0f,  h), up, glm::vec2(t, t) });
    vertices.push_back({ glm::vec3( h, 0.0f, -h), up, glm::vec2(t, 0.0f) });

    const uint16_t floor_indices[FLOOR_INDEX_COUNT] = { 0, 1, 2, 0, 2, 3 };
    for (uint16_t index : floor_indices) {
        indices.push_back(base + index);
    }
}

void generateCone(std::vector<Vertex>& vertices) {
    const uint32_t spokes = 8;
    const uint32_t segments = 32;
    const glm::vec3 no_normal(0.0f);
    const glm::vec2 no_uv(0.0f);

    auto addLine = [&](const glm::vec3& a, const glm::vec3& b) {
        vertices.push_back({ a, no_normal, no_uv });
        vertices.push_back({ b, no_normal, no_uv });
    };

    addLine(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    for (uint32_t i = 0; i < spokes; ++i) {
        float a = 2.0f * glm::pi<float>() * float(i) / float(spokes);
        addLine(glm::vec3(0.0f), glm::vec3(std::cos(a), std::sin(a), 1.0f));
    }

    for (uint32_t i = 0; i < segments; ++i) {
        float a0 = 2.0f * glm::pi<float>() * float(i) / float(segments);
        float a1 = 2.0f * glm::pi<float>() * float(i + 1) / float(segments);
        addLine(glm::vec3(std::cos(a0), std::sin(a0), 1.0f),
                glm::vec3(std::cos(a1), std::sin(a1), 1.0f));
    }
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                  VkBuffer& buffer, VmaAllocation& allocation, void** mapped_ptr = nullptr) {
    auto& context = graphics::internal::context;
    VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VmaAllocationCreateInfo allocInfo = { .usage = VMA_MEMORY_USAGE_CPU_TO_GPU };

    if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr) != VK_SUCCESS) {
        return false;
    }
    if (mapped_ptr) vmaMapMemory(context.allocator, allocation, mapped_ptr);
    return true;
}

void writeBuffer(void* mapped, VmaAllocation allocation, const void* data, size_t size) {
    std::memcpy(mapped, data, size);
    vmaFlushAllocation(graphics::internal::context.allocator, allocation, 0, VK_WHOLE_SIZE);
}

VkCommandPool upload_command_pool = VK_NULL_HANDLE;

VkCommandBuffer beginUploadCommands() {
    auto& context = graphics::internal::context;
    VkCommandBufferAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = upload_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(context.device, &allocate_info, &command_buffer);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(command_buffer, &begin_info);
    return command_buffer;
}

void endUploadCommands(VkCommandBuffer command_buffer) {
    auto& context = graphics::internal::context;
    vkEndCommandBuffer(command_buffer);

    VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(context.device, &fence_info, nullptr, &fence);

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };
    vkQueueSubmit(context.graphics_queue, 1, &submit_info, fence);
    vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(context.device, fence, nullptr);
    vkFreeCommandBuffers(context.device, upload_command_pool, 1, &command_buffer);
}

void createStagingBuffer(const void* data, VkDeviceSize size, VkBuffer& staging, VmaAllocation& staging_alloc) {
    auto& context = graphics::internal::context;
    void* mapped = nullptr;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging, staging_alloc, &mapped);
    writeBuffer(mapped, staging_alloc, data, size);
    vmaUnmapMemory(context.allocator, staging_alloc);
}

void createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                             VkBuffer& buffer, VmaAllocation& allocation) {
    auto& context = graphics::internal::context;

    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VmaAllocationCreateInfo alloc_info = { .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
    vmaCreateBuffer(context.allocator, &buffer_info, &alloc_info, &buffer, &allocation, nullptr);

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = VK_NULL_HANDLE;
    createStagingBuffer(data, size, staging, staging_alloc);

    VkCommandBuffer command_buffer = beginUploadCommands();
    VkBufferCopy region = { .srcOffset = 0, .dstOffset = 0, .size = size };
    vkCmdCopyBuffer(command_buffer, staging, buffer, 1, &region);
    endUploadCommands(command_buffer);

    vmaDestroyBuffer(context.allocator, staging, staging_alloc);
}

Texture createTexture(const unsigned char* rgba, uint32_t width, uint32_t height, VkFormat format) {
    auto& context = graphics::internal::context;
    Texture texture = {};

    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = { width, height, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VmaAllocationCreateInfo alloc_info = { .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
    vmaCreateImage(context.allocator, &image_info, &alloc_info, &texture.image, &texture.allocation, nullptr);

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = VK_NULL_HANDLE;
    createStagingBuffer(rgba, VkDeviceSize(width) * height * 4, staging, staging_alloc);

    VkImageSubresourceRange subresource_range = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    VkCommandBuffer command_buffer = beginUploadCommands();

    VkImageMemoryBarrier to_transfer = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = texture.image,
        .subresourceRange = subresource_range,
    };
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &to_transfer);

    VkBufferImageCopy region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { width, height, 1 },
    };
    vkCmdCopyBufferToImage(command_buffer, staging, texture.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkImageMemoryBarrier to_shader_read = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = texture.image,
        .subresourceRange = subresource_range,
    };
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &to_shader_read);

    endUploadCommands(command_buffer);
    vmaDestroyBuffer(context.allocator, staging, staging_alloc);

    VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = texture.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = subresource_range,
    };
    vkCreateImageView(context.device, &view_info, nullptr, &texture.view);
    return texture;
}

ShadowMap createShadowMap(uint32_t size, VkFormat format) {
    auto& context = graphics::internal::context;
    ShadowMap sm = {};

    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = { size, size, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VmaAllocationCreateInfo alloc_info = { .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
    vmaCreateImage(context.allocator, &image_info, &alloc_info, &sm.image, &sm.allocation, nullptr);

    VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = sm.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCreateImageView(context.device, &view_info, nullptr, &sm.view);

    createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 sm.ubo_buffer, sm.ubo_alloc, &sm.ubo_mapped);
    return sm;
}

void destroyShadowMap(ShadowMap& sm) {
    auto& context = graphics::internal::context;
    if (sm.view != VK_NULL_HANDLE) {
        vkDestroyImageView(context.device, sm.view, nullptr);
    }
    if (sm.image != VK_NULL_HANDLE) {
        vmaDestroyImage(context.allocator, sm.image, sm.allocation);
    }
    if (sm.ubo_buffer != VK_NULL_HANDLE) {
        vmaUnmapMemory(context.allocator, sm.ubo_alloc);
        vmaDestroyBuffer(context.allocator, sm.ubo_buffer, sm.ubo_alloc);
    }
    sm = {};
}

bool decodePng(const std::string& filename, std::vector<unsigned char>& pixels,
               unsigned& width, unsigned& height) {
    const std::string directories[] = { "", "textures/", "../textures/", "../../textures/" };
    for (const std::string& directory : directories) {
        pixels.clear();
        if (lodepng::decode(pixels, width, height, directory + filename, LCT_RGBA, 8) == 0) {
            return true;
        }
    }
    return false;
}

VkImageView loadTexture(const char* filename, VkFormat format, const unsigned char fallback_rgba[4]) {
    std::vector<unsigned char> pixels;
    unsigned width = 1;
    unsigned height = 1;

    if (!decodePng(filename, pixels, width, height)) {
        std::cerr << "Failed to load texture: " << filename << '\n';
        pixels.assign(fallback_rgba, fallback_rgba + 4);
        width = 1;
        height = 1;
    }

    textures.push_back(createTexture(pixels.data(), width, height, format));
    return textures.back().view;
}

void writeModel(uint32_t slot, const glm::mat4& model, const glm::vec3& albedo,
                const glm::vec3& specular, float shininess, bool unlit) {
    ModelUniforms uniforms = {};
    uniforms.model = model;
    uniforms.normal_matrix = glm::transpose(glm::inverse(model));
    uniforms.albedo = albedo;
    uniforms.specular = specular;
    uniforms.shininess = shininess;
    uniforms.unlit = unlit ? 1u : 0u;
    std::memcpy(model_ubo_mapped + slot * model_ubo_stride, &uniforms, sizeof(uniforms));
}

glm::mat4 coneModelMatrix(const glm::vec3& position, const glm::vec3& direction,
                          float angle_radians, float length) {
    glm::vec3 z = safeNormalize(direction, glm::vec3(0.0f, -1.0f, 0.0f));
    glm::vec3 helper = std::abs(z.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 x = glm::normalize(glm::cross(helper, z));
    glm::vec3 y = glm::cross(z, x);
    float radius = std::tan(angle_radians) * length;

    glm::mat4 m(1.0f);
    m[0] = glm::vec4(x * radius, 0.0f);
    m[1] = glm::vec4(y * radius, 0.0f);
    m[2] = glm::vec4(z * length, 0.0f);
    m[3] = glm::vec4(position, 1.0f);
    return m;
}

VkPipeline createPipeline(VkShaderModule vert_module, VkShaderModule frag_module,
                          VkPrimitiveTopology topology) {
    auto& context = graphics::internal::context;

    VkPipelineShaderStageCreateInfo shader_stages[] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert_module, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag_module, .pName = "main" }
    };

    VkVertexInputBindingDescription binding_desc = { .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription attribute_descs[] = {
        { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
        { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
        { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv) },
    };
    VkPipelineVertexInputStateCreateInfo vertex_input_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding_desc,
        .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attribute_descs,
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = topology };
    VkPipelineViewportStateCreateInfo viewport_state = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1 };

    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo multisampling = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineDepthStencilStateCreateInfo depth_stencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL
    };
    VkPipelineColorBlendAttachmentState color_blend_attachment = { .blendEnable = VK_FALSE, .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT };
    VkPipelineColorBlendStateCreateInfo color_blending = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &color_blend_attachment };

    VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic_state_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dynamic_states };

    VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = shader_stages,
        .pVertexInputState = &vertex_input_info, .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state, .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling, .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blending, .pDynamicState = &dynamic_state_info,
        .layout = pipeline_layout, .renderPass = context.render_pass,
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
    return pipeline;
}

VkPipeline createShadowPipeline(VkShaderModule shadow_vert_module, VkFormat depth_format, bool position_only_attributes) {
    auto& context = graphics::internal::context;

    VkPipelineShaderStageCreateInfo stage_infos[1] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = shadow_vert_module,
            .pName = "main",
        }
    };

    VkVertexInputBindingDescription buffer_binding = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    VkVertexInputAttributeDescription attributes[] = {
        { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
        { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
        { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv) },
    };

    VkPipelineVertexInputStateCreateInfo input_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &buffer_binding,
        .vertexAttributeDescriptionCount = position_only_attributes ? 1u : 3u,
        .pVertexAttributeDescriptions = attributes,
    };

    VkPipelineInputAssemblyStateCreateInfo assembly_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkPipelineViewportStateCreateInfo viewport_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    VkPipelineRasterizationStateCreateInfo raster_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_TRUE,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo sample_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    VkPipelineDepthStencilStateCreateInfo depth_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
    };

    VkPipelineColorBlendStateCreateInfo blend_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 0,
        .pAttachments = nullptr,
    };

    VkDynamicState dyn_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
    };

    VkPipelineDynamicStateCreateInfo dyn_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = sizeof(dyn_states) / sizeof(dyn_states[0]),
        .pDynamicStates = dyn_states,
    };

    VkPipelineRenderingCreateInfoKHR format_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
        .colorAttachmentCount = 0,
        .pColorAttachmentFormats = nullptr,
        .depthAttachmentFormat = depth_format,
    };

    VkGraphicsPipelineCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &format_info,
        .stageCount = 1,
        .pStages = stage_infos,
        .pVertexInputState = &input_state_info,
        .pInputAssemblyState = &assembly_state_info,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_info,
        .pMultisampleState = &sample_info,
        .pDepthStencilState = &depth_info,
        .pColorBlendState = &blend_info,
        .pDynamicState = &dyn_state_info,
        .layout = pipeline_layout,
        .renderPass = VK_NULL_HANDLE,
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    return pipeline;
}

void objectMaterialUI(ObjectSettings& object) {
    ImGui::Combo("Material", &object.material, material_names.data(), static_cast<int>(material_names.size()));
    ImGui::Combo("Sampling effect", &object.effect, EFFECT_NAMES, IM_ARRAYSIZE(EFFECT_NAMES));
    if (object.effect != EFFECT_NONE) {
        ImGui::SliderFloat("Effect strength", &object.effect_strength, 0.0f, 3.0f);
    }
    ImGui::ColorEdit3("Albedo tint", glm::value_ptr(object.albedo_tint));
    ImGui::ColorEdit3("Specular tint", glm::value_ptr(object.specular_tint));
    ImGui::SliderFloat("Shininess", &object.shininess, 1.0f, 512.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
}

void cameraUI() {
    ImGui::Checkbox("Perspective Projection", &use_perspective);
    if (use_perspective) ImGui::SliderFloat("FOV", &fov, 20.0f, 120.0f);
    else ImGui::SliderFloat("Ortho Size", &ortho_size, 0.5f, 10.0f);
    ImGui::DragFloat3("Camera Position", glm::value_ptr(camera_pos), 0.05f);
    ImGui::SliderFloat("Camera Yaw", &camera_yaw, -180.0f, 180.0f);
    ImGui::SliderFloat("Camera Pitch", &camera_pitch, -89.0f, 89.0f);
}

void ambientAndDirectionalUI() {
    ImGui::ColorEdit3("Ambient light", glm::value_ptr(ambient_light));

    ImGui::Separator();
    ImGui::Checkbox("Directional light", &dir_light_enabled);
    ImGui::DragFloat3("Direction", glm::value_ptr(dir_light_direction), 0.01f);
    ImGui::ColorEdit3("Dir. color", glm::value_ptr(dir_light_color));
    ImGui::DragFloat("Dir. intensity", &dir_light_intensity, 0.01f, 0.0f, 100.0f);
}

void shadowMappingUI() {
    ImGui::Checkbox("Directional light shadow", &dir_shadow_enabled);
    ImGui::SliderFloat("Ortho projection size (area)", &shadow_ortho_size, 1.5f, 20.0f, "%.2f");
    ImGui::SliderFloat("Ortho near plane", &shadow_near_plane, 0.0f, 5.0f, "%.2f");
    ImGui::SliderFloat("Ortho far plane", &shadow_far_plane, 5.0f, 80.0f, "%.1f");
    ImGui::SliderFloat("Sun LookAt distance", &shadow_light_distance, 2.0f, 35.0f, "%.1f");
    ImGui::Checkbox("Follow camera position", &shadow_follow_camera);

    ImGui::Separator();
    ImGui::SliderFloat("Depth bias constant", &depth_bias_constant, 0.0f, 10.0f, "%.2f");
    ImGui::SliderFloat("Depth bias slope", &depth_bias_slope, 0.0f, 10.0f, "%.2f");

    ImGui::Separator();
    ImGui::Checkbox("Spot light shadow (Bonus)", &spot_shadow_enabled);
    if (spot_shadow_enabled && !spot_lights.empty()) {
        int max_idx = static_cast<int>(spot_lights.size()) - 1;
        shadow_spot_index = std::clamp(shadow_spot_index, 0, max_idx);
        if (max_idx > 0) {
            ImGui::SliderInt("Shadow spot light #", &shadow_spot_index, 0, max_idx);
        }
        ImGui::SliderFloat("Spot shadow near", &spot_shadow_near, 0.05f, 2.0f, "%.2f");
        ImGui::SliderFloat("Spot shadow far", &spot_shadow_far, 5.0f, 50.0f, "%.1f");
    }
}

void pointLightsUI() {
    if (point_lights.size() < MAX_POINT_LIGHTS) {
        if (ImGui::Button("Add point light")) {
            point_lights.push_back({ glm::vec3(0.0f, 2.0f, 2.0f), glm::vec3(1.0f), 4.0f });
        }
    } else {
        ImGui::TextDisabled("Point light limit reached (%u)", MAX_POINT_LIGHTS);
    }

    int remove_index = -1;
    for (size_t i = 0; i < point_lights.size(); ++i) {
        PointLightSettings& light = point_lights[i];
        ImGui::PushID(static_cast<int>(i));
        std::string title = "Point light " + std::to_string(i + 1);
        if (ImGui::TreeNodeEx(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Position", glm::value_ptr(light.position), 0.05f);
            ImGui::ColorEdit3("Color", glm::value_ptr(light.color));
            ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 1000.0f);
            if (ImGui::Button("Remove")) remove_index = static_cast<int>(i);
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (remove_index >= 0) point_lights.erase(point_lights.begin() + remove_index);
}

void spotLightsUI() {
    if (spot_lights.size() < MAX_SPOT_LIGHTS) {
        if (ImGui::Button("Add spot light")) {
            spot_lights.push_back({ glm::vec3(0.0f, 4.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f),
                                    glm::vec3(1.0f), 12.0f, 15.0f, 25.0f });
        }
    } else {
        ImGui::TextDisabled("Spot light limit reached (%u)", MAX_SPOT_LIGHTS);
    }

    int remove_index = -1;
    for (size_t i = 0; i < spot_lights.size(); ++i) {
        SpotLightSettings& light = spot_lights[i];
        ImGui::PushID(static_cast<int>(i));
        std::string title = "Spot light " + std::to_string(i + 1);
        if (ImGui::TreeNodeEx(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Position", glm::value_ptr(light.position), 0.05f);
            ImGui::DragFloat3("Direction", glm::value_ptr(light.direction), 0.01f);
            ImGui::ColorEdit3("Color", glm::value_ptr(light.color));
            ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 1000.0f);
            ImGui::SliderFloat("Outer angle", &light.outer_angle, 1.0f, 80.0f, "%.1f deg");
            light.inner_angle = std::min(light.inner_angle, light.outer_angle);
            ImGui::SliderFloat("Inner angle", &light.inner_angle, 0.0f, light.outer_angle, "%.1f deg");
            if (ImGui::Button("Remove")) remove_index = static_cast<int>(i);
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (remove_index >= 0) spot_lights.erase(spot_lights.begin() + remove_index);
}

void objectsUI() {
    ImGui::Checkbox("Play Animation", &anim_playing);
    ImGui::SliderFloat("Speed", &anim_speed, 0.0f, 3.0f);

    if (ImGui::TreeNodeEx("Sphere 1", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Position", glm::value_ptr(obj1_pos), 0.05f);
        ImGui::SliderFloat3("Rotation", glm::value_ptr(obj1_rot), -180.0f, 180.0f);
        ImGui::DragFloat3("Scale", glm::value_ptr(obj1_scale), 0.05f, 0.1f, 5.0f);
        ImGui::Checkbox("Animate along Trefoil Knot", &use_trajectory);
        if (use_trajectory) ImGui::SliderFloat("Radius", &traj_radius, 0.5f, 5.0f);
        objectMaterialUI(objects[OBJECT_SPHERE_1]);
        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Sphere 2", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Render Object 2", &obj2_enabled);
        objectMaterialUI(objects[OBJECT_SPHERE_2]);
        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Floor", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Render Floor", &floor_enabled);
        ImGui::DragFloat("Floor Y", &floor_y, 0.05f);
        objectMaterialUI(objects[OBJECT_FLOOR]);
        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Back Wall", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Render Back Wall", &wall_enabled);
        ImGui::DragFloat("Wall Z", &wall_z, 0.05f);
        objectMaterialUI(objects[OBJECT_WALL]);
        ImGui::TreePop();
    }
}

void userInterface() {
    ImGui::SetNextWindowSize(ImVec2(430.0f, 760.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lab 4: Shadow mapping (Variant 5, Sphere)");

    if (ImGui::CollapsingHeader("Shadow mapping", ImGuiTreeNodeFlags_DefaultOpen)) shadowMappingUI();
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) cameraUI();
    if (ImGui::CollapsingHeader("Ambient and directional light", ImGuiTreeNodeFlags_DefaultOpen)) ambientAndDirectionalUI();

    ImGui::Checkbox("Show light sources", &show_lights);
    if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) pointLightsUI();
    if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) spotLightsUI();

    if (ImGui::CollapsingHeader("Objects, textures and effects", ImGuiTreeNodeFlags_DefaultOpen)) objectsUI();

    ImGui::End();
}

} // namespace

bool initialize() {
    auto& context = graphics::internal::context;

    VkPhysicalDevice physical_device = getPhysicalDevice();
    shadow_depth_format = findDepthFormat(physical_device);

    VkCommandPoolCreateInfo command_pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = findGraphicsQueueFamily(physical_device),
    };
    vkCreateCommandPool(context.device, &command_pool_info, nullptr, &upload_command_pool);

    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
    generateSphere(vertices, indices);
    sphere_index_count = static_cast<uint32_t>(indices.size());
    floor_first_index = sphere_index_count;
    generateFloor(vertices, indices);

    createDeviceLocalBuffer(vertices.data(), sizeof(Vertex) * vertices.size(),
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer, vertex_buffer_alloc);
    createDeviceLocalBuffer(indices.data(), sizeof(uint16_t) * indices.size(),
                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer, index_buffer_alloc);

    std::vector<Vertex> cone_vertices;
    generateCone(cone_vertices);
    cone_vertex_count = static_cast<uint32_t>(cone_vertices.size());
    createDeviceLocalBuffer(cone_vertices.data(), sizeof(Vertex) * cone_vertices.size(),
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, cone_vertex_buffer, cone_vertex_buffer_alloc);

    createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 scene_ubo_buffer, scene_ubo_alloc, &scene_ubo_mapped);
    createBuffer(sizeof(GpuPointLight) * MAX_POINT_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 point_lights_buffer, point_lights_alloc, &point_lights_mapped);
    createBuffer(sizeof(GpuSpotLight) * MAX_SPOT_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 spot_lights_buffer, spot_lights_alloc, &spot_lights_mapped);

    dir_shadow = createShadowMap(SHADOW_MAP_SIZE, shadow_depth_format);
    spot_shadow = createShadowMap(SHADOW_MAP_SIZE, shadow_depth_format);

    VkPhysicalDeviceProperties device_properties = {};
    vkGetPhysicalDeviceProperties(physical_device, &device_properties);
    model_ubo_stride = alignUp(sizeof(ModelUniforms), device_properties.limits.minUniformBufferOffsetAlignment);

    void* model_ubo_ptr = nullptr;
    createBuffer(model_ubo_stride * MODEL_COUNT, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 model_ubo_buffer, model_ubo_alloc, &model_ubo_ptr);
    model_ubo_mapped = static_cast<uint8_t*>(model_ubo_ptr);

    VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    vkCreateSampler(context.device, &sampler_info, nullptr, &texture_sampler);

    VkSamplerCreateInfo shadow_sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_TRUE,
        .compareOp = VK_COMPARE_OP_LESS,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
        .unnormalizedCoordinates = VK_FALSE,
    };
    vkCreateSampler(context.device, &shadow_sampler_info, nullptr, &shadow_sampler);

    const unsigned char black_pixel[4] = { 0, 0, 0, 255 };
    const unsigned char missing_pixel[4] = { 255, 0, 255, 255 };
    textures.push_back(createTexture(black_pixel, 1, 1, VK_FORMAT_R8G8B8A8_UNORM));
    VkImageView black_view = textures.back().view;

    VkImageView material_views[MATERIAL_COUNT][3];
    for (uint32_t m = 0; m < MATERIAL_COUNT; ++m) {
        const MaterialDefinition& definition = MATERIAL_DEFINITIONS[m];
        material_names.push_back(definition.name);
        material_views[m][0] = loadTexture(definition.albedo, VK_FORMAT_R8G8B8A8_SRGB, missing_pixel);
        material_views[m][1] = definition.specular
            ? loadTexture(definition.specular, VK_FORMAT_R8G8B8A8_UNORM, black_pixel) : black_view;
        material_views[m][2] = definition.emissive
            ? loadTexture(definition.emissive, VK_FORMAT_R8G8B8A8_SRGB, black_pixel) : black_view;
    }

    vkDestroyCommandPool(context.device, upload_command_pool, nullptr);
    upload_command_pool = VK_NULL_HANDLE;

    VkDescriptorSetLayoutBinding scene_bindings[] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
    };
    VkDescriptorSetLayoutCreateInfo scene_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 5,
        .pBindings = scene_bindings,
    };
    vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr, &scene_set_layout);

    VkDescriptorSetLayoutBinding material_bindings[] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
    };
    VkDescriptorSetLayoutCreateInfo material_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = material_bindings,
    };
    vkCreateDescriptorSetLayout(context.device, &material_layout_info, nullptr, &material_set_layout);

    VkDescriptorPoolSize pool_sizes[] = {
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 3 },
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = MATERIAL_COUNT },
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 6 },
        { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 3 * MATERIAL_COUNT + 6 },
    };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 3 + MATERIAL_COUNT,
        .poolSizeCount = 4,
        .pPoolSizes = pool_sizes,
    };
    vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool);

    VkDescriptorSetLayout scene_layouts[3] = { scene_set_layout, scene_set_layout, scene_set_layout };
    VkDescriptorSet allocated_scene_sets[3] = {};
    VkDescriptorSetAllocateInfo scene_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 3,
        .pSetLayouts = scene_layouts,
    };
    vkAllocateDescriptorSets(context.device, &scene_alloc_info, allocated_scene_sets);
    scene_set = allocated_scene_sets[0];
    dir_shadow.scene_set = allocated_scene_sets[1];
    spot_shadow.scene_set = allocated_scene_sets[2];

    std::vector<VkDescriptorSetLayout> material_layouts(MATERIAL_COUNT, material_set_layout);
    VkDescriptorSetAllocateInfo material_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = MATERIAL_COUNT,
        .pSetLayouts = material_layouts.data(),
    };
    vkAllocateDescriptorSets(context.device, &material_alloc_info, material_sets);

    VkDescriptorBufferInfo scene_buffer_info = { .buffer = scene_ubo_buffer, .offset = 0, .range = sizeof(SceneUniforms) };
    VkDescriptorBufferInfo dir_shadow_ubo_info = { .buffer = dir_shadow.ubo_buffer, .offset = 0, .range = sizeof(SceneUniforms) };
    VkDescriptorBufferInfo spot_shadow_ubo_info = { .buffer = spot_shadow.ubo_buffer, .offset = 0, .range = sizeof(SceneUniforms) };
    VkDescriptorBufferInfo point_buffer_info = { .buffer = point_lights_buffer, .offset = 0, .range = sizeof(GpuPointLight) * MAX_POINT_LIGHTS };
    VkDescriptorBufferInfo spot_buffer_info = { .buffer = spot_lights_buffer, .offset = 0, .range = sizeof(GpuSpotLight) * MAX_SPOT_LIGHTS };

    VkDescriptorImageInfo dir_shadow_image_info = {
        .sampler = shadow_sampler,
        .imageView = dir_shadow.view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkDescriptorImageInfo spot_shadow_image_info = {
        .sampler = shadow_sampler,
        .imageView = spot_shadow.view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkWriteDescriptorSet scene_writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = scene_set, .dstBinding = 0,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &scene_buffer_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = scene_set, .dstBinding = 1,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &point_buffer_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = scene_set, .dstBinding = 2,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &spot_buffer_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = scene_set, .dstBinding = 3,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &dir_shadow_image_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = scene_set, .dstBinding = 4,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &spot_shadow_image_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dir_shadow.scene_set, .dstBinding = 0,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &dir_shadow_ubo_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = spot_shadow.scene_set, .dstBinding = 0,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &spot_shadow_ubo_info },
    };
    vkUpdateDescriptorSets(context.device, sizeof(scene_writes) / sizeof(scene_writes[0]), scene_writes, 0, nullptr);

    for (uint32_t m = 0; m < MATERIAL_COUNT; ++m) {
        VkDescriptorBufferInfo model_buffer_info = { .buffer = model_ubo_buffer, .offset = 0, .range = sizeof(ModelUniforms) };
        VkDescriptorImageInfo image_infos[3];
        for (uint32_t k = 0; k < 3; ++k) {
            image_infos[k] = {
                .sampler = texture_sampler,
                .imageView = material_views[m][k],
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
        }
        VkWriteDescriptorSet material_writes[] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = material_sets[m], .dstBinding = 0,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &model_buffer_info },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = material_sets[m], .dstBinding = 1,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &image_infos[0] },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = material_sets[m], .dstBinding = 2,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &image_infos[1] },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = material_sets[m], .dstBinding = 3,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &image_infos[2] },
        };
        vkUpdateDescriptorSets(context.device, 4, material_writes, 0, nullptr);
    }

    VkDescriptorSetLayout set_layouts[] = { scene_set_layout, material_set_layout };
    VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = set_layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range,
    };
    vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr, &pipeline_layout);

    auto vert_code = readShaderFile("shader.vert.spv");
    auto frag_code = readShaderFile("shader.frag.spv");
    auto shadow_vert_code = readShaderFile("shadow.vert.spv", true);
    if (shadow_vert_code.empty()) {
        shadow_vert_code = readShaderFile("shadowmapping.vert.spv", true);
    }
    const bool has_dedicated_shadow_vert = !shadow_vert_code.empty();

    VkShaderModule vert_module = createShaderModule(context.device, vert_code);
    VkShaderModule frag_module = createShaderModule(context.device, frag_code);
    VkShaderModule shadow_vert_module = has_dedicated_shadow_vert
        ? createShaderModule(context.device, shadow_vert_code)
        : vert_module;

    triangle_pipeline = createPipeline(vert_module, frag_module, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    line_pipeline = createPipeline(vert_module, frag_module, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
    shadow_pipeline = createShadowPipeline(shadow_vert_module, shadow_depth_format, has_dedicated_shadow_vert);

    if (has_dedicated_shadow_vert) {
        vkDestroyShaderModule(context.device, shadow_vert_module, nullptr);
    }
    vkDestroyShaderModule(context.device, frag_module, nullptr);
    vkDestroyShaderModule(context.device, vert_module, nullptr);
    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);
    vkDestroyPipeline(context.device, shadow_pipeline, nullptr);
    vkDestroyPipeline(context.device, line_pipeline, nullptr);
    vkDestroyPipeline(context.device, triangle_pipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
    vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, material_set_layout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);

    destroyShadowMap(spot_shadow);
    destroyShadowMap(dir_shadow);

    for (const Texture& texture : textures) {
        vkDestroyImageView(context.device, texture.view, nullptr);
        vmaDestroyImage(context.allocator, texture.image, texture.allocation);
    }
    textures.clear();
    vkDestroySampler(context.device, shadow_sampler, nullptr);
    vkDestroySampler(context.device, texture_sampler, nullptr);

    vmaUnmapMemory(context.allocator, model_ubo_alloc);
    vmaDestroyBuffer(context.allocator, model_ubo_buffer, model_ubo_alloc);
    vmaUnmapMemory(context.allocator, spot_lights_alloc);
    vmaDestroyBuffer(context.allocator, spot_lights_buffer, spot_lights_alloc);
    vmaUnmapMemory(context.allocator, point_lights_alloc);
    vmaDestroyBuffer(context.allocator, point_lights_buffer, point_lights_alloc);
    vmaUnmapMemory(context.allocator, scene_ubo_alloc);
    vmaDestroyBuffer(context.allocator, scene_ubo_buffer, scene_ubo_alloc);

    vmaDestroyBuffer(context.allocator, cone_vertex_buffer, cone_vertex_buffer_alloc);
    vmaDestroyBuffer(context.allocator, index_buffer, index_buffer_alloc);
    vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_buffer_alloc);
}

void update(double time) {
    if (last_time == 0.0) last_time = time;
    float dt = static_cast<float>(time - last_time);
    last_time = time;
    if (anim_playing) anim_time += dt * anim_speed;

    userInterface();

    auto& context = graphics::internal::context;
    float aspect = context.swapchain_extent.height > 0
        ? (float)context.swapchain_extent.width / (float)context.swapchain_extent.height
        : 1.0f;

    glm::mat4 proj = use_perspective
        ? glm::perspective(glm::radians(fov), aspect, near_plane, far_plane)
        : glm::ortho(-ortho_size * aspect, ortho_size * aspect, -ortho_size, ortho_size, near_plane, far_plane);
    proj[1][1] *= -1.0f;

    glm::mat4 view = glm::rotate(glm::mat4(1.0f), glm::radians(-camera_pitch), glm::vec3(1.0f, 0.0f, 0.0f));
    view = glm::rotate(view, glm::radians(-camera_yaw), glm::vec3(0.0f, 1.0f, 0.0f));
    view = glm::translate(view, -camera_pos);

    glm::vec3 norm_dir_light = safeNormalize(dir_light_direction, glm::vec3(0.0f, -1.0f, 0.0f));
    glm::vec3 shadow_target = shadow_follow_camera ? camera_pos : glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 shadow_eye = shadow_target - norm_dir_light * shadow_light_distance;
    glm::mat4 dir_light_view = glm::lookAt(shadow_eye, shadow_target, chooseUpVector(norm_dir_light));
    glm::mat4 dir_light_proj = glm::ortho(
        -shadow_ortho_size, shadow_ortho_size,
        -shadow_ortho_size, shadow_ortho_size,
        shadow_near_plane, shadow_far_plane);
    dir_light_proj[1][1] *= -1.0f;
    glm::mat4 dir_shadow_matrix = dir_light_proj * dir_light_view;

    int active_spot_idx = -1;
    glm::mat4 spot_shadow_matrix(1.0f);
    if (spot_shadow_enabled && !spot_lights.empty()) {
        active_spot_idx = std::clamp(shadow_spot_index, 0, static_cast<int>(spot_lights.size()) - 1);
        const SpotLightSettings& spot = spot_lights[active_spot_idx];
        glm::vec3 spot_dir = safeNormalize(spot.direction, glm::vec3(0.0f, -1.0f, 0.0f));
        float spot_fov = glm::radians(std::clamp(2.0f * spot.outer_angle, 2.0f, 170.0f));
        glm::mat4 spot_view = glm::lookAt(spot.position, spot.position + spot_dir, chooseUpVector(spot_dir));
        glm::mat4 spot_proj = glm::perspective(spot_fov, 1.0f, spot_shadow_near, spot_shadow_far);
        spot_proj[1][1] *= -1.0f;
        spot_shadow_matrix = spot_proj * spot_view;
    }

    SceneUniforms dir_shadow_ubo = {};
    dir_shadow_ubo.view_projection = dir_shadow_matrix;
    writeBuffer(dir_shadow.ubo_mapped, dir_shadow.ubo_alloc, &dir_shadow_ubo, sizeof(dir_shadow_ubo));

    SceneUniforms spot_shadow_ubo = {};
    spot_shadow_ubo.view_projection = spot_shadow_matrix;
    writeBuffer(spot_shadow.ubo_mapped, spot_shadow.ubo_alloc, &spot_shadow_ubo, sizeof(spot_shadow_ubo));

    SceneUniforms scene = {};
    scene.view_projection = proj * view;
    scene.dir_shadow_projection = dir_shadow_matrix;
    scene.spot_shadow_projection = spot_shadow_matrix;
    scene.camera_position = camera_pos;
    scene.directional_light_direction = norm_dir_light;
    scene.directional_light_color = dir_light_enabled ? dir_light_color * dir_light_intensity : glm::vec3(0.0f);
    scene.ambient_light = ambient_light;
    scene.point_light_count = static_cast<uint32_t>(point_lights.size());
    scene.spot_light_count = static_cast<uint32_t>(spot_lights.size());
    scene.shadow_spot_index = active_spot_idx;
    scene.dir_shadow_enabled = (dir_light_enabled && dir_shadow_enabled) ? 1u : 0u;
    writeBuffer(scene_ubo_mapped, scene_ubo_alloc, &scene, sizeof(scene));

    std::vector<GpuPointLight> gpu_point_lights;
    for (const PointLightSettings& light : point_lights) {
        GpuPointLight gpu = {};
        gpu.position = light.position;
        gpu.color = light.color * light.intensity;
        gpu_point_lights.push_back(gpu);
    }
    if (!gpu_point_lights.empty()) {
        writeBuffer(point_lights_mapped, point_lights_alloc, gpu_point_lights.data(),
                    sizeof(GpuPointLight) * gpu_point_lights.size());
    }

    std::vector<GpuSpotLight> gpu_spot_lights;
    for (const SpotLightSettings& light : spot_lights) {
        GpuSpotLight gpu = {};
        gpu.position = light.position;
        gpu.color = light.color * light.intensity;
        gpu.direction = safeNormalize(light.direction, glm::vec3(0.0f, -1.0f, 0.0f));
        gpu.inner_cos = std::cos(glm::radians(light.inner_angle));
        gpu.outer_cos = std::cos(glm::radians(light.outer_angle));
        gpu_spot_lights.push_back(gpu);
    }
    if (!gpu_spot_lights.empty()) {
        writeBuffer(spot_lights_mapped, spot_lights_alloc, gpu_spot_lights.data(),
                    sizeof(GpuSpotLight) * gpu_spot_lights.size());
    }

    float t = static_cast<float>(anim_time);
    if (use_trajectory) {
        obj1_pos.x = traj_radius * (std::sin(t) + 2.0f * std::sin(2.0f * t)) * 0.3f;
        obj1_pos.y = traj_radius * (std::cos(t) - 2.0f * std::cos(2.0f * t)) * 0.3f;
        obj1_pos.z = traj_radius * (-std::sin(3.0f * t)) * 0.3f;
        if (anim_playing) {
            obj1_rot.y += dt * anim_speed * 50.0f;
        }
    }

    glm::mat4 model1 = glm::translate(glm::mat4(1.0f), obj1_pos);
    model1 = glm::rotate(model1, glm::radians(obj1_rot.y), glm::vec3(0, 1, 0));
    model1 = glm::rotate(model1, glm::radians(obj1_rot.x), glm::vec3(1, 0, 0));
    model1 = glm::rotate(model1, glm::radians(obj1_rot.z), glm::vec3(0, 0, 1));
    model1 = glm::scale(model1, obj1_scale);

    glm::vec3 pos2(2.5f * std::cos(t), 0.0f, 2.5f * std::sin(t));
    glm::mat4 model2 = glm::translate(glm::mat4(1.0f), pos2);
    model2 = glm::scale(model2, obj2_scale);

    glm::mat4 model_floor = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, floor_y, 0.0f));

    glm::mat4 model_wall = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, floor_y + FLOOR_HALF_SIZE, wall_z));
    model_wall = glm::rotate(model_wall, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    const glm::mat4 scene_models[SCENE_OBJECT_COUNT] = { model1, model2, model_floor, model_wall };
    for (uint32_t i = 0; i < SCENE_OBJECT_COUNT; ++i) {
        const ObjectSettings& object = objects[i];
        writeModel(i, scene_models[i], object.albedo_tint, object.specular_tint, object.shininess, false);
    }

    if (show_lights) {
        for (size_t i = 0; i < point_lights.size(); ++i) {
            glm::mat4 marker = glm::translate(glm::mat4(1.0f), point_lights[i].position);
            marker = glm::scale(marker, glm::vec3(LIGHT_MARKER_SCALE));
            writeModel(POINT_MARKER_BASE + static_cast<uint32_t>(i), marker,
                       point_lights[i].color, glm::vec3(0.0f), 1.0f, true);
        }
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            const SpotLightSettings& light = spot_lights[i];
            glm::mat4 marker = glm::translate(glm::mat4(1.0f), light.position);
            marker = glm::scale(marker, glm::vec3(LIGHT_MARKER_SCALE));
            writeModel(SPOT_MARKER_BASE + static_cast<uint32_t>(i), marker,
                       light.color, glm::vec3(0.0f), 1.0f, true);

            glm::mat4 cone = coneModelMatrix(light.position, light.direction,
                                             glm::radians(light.outer_angle), SPOT_CONE_LENGTH);
            writeModel(SPOT_CONE_BASE + static_cast<uint32_t>(i), cone,
                       glm::vec3(1.0f), glm::vec3(0.0f), 1.0f, true);
        }
    }

    vmaFlushAllocation(context.allocator, model_ubo_alloc, 0, VK_WHOLE_SIZE);
}

void render(const graphics::internal::FrameData& fd) {
    auto& context = graphics::internal::context;
    vkResetCommandBuffer(fd.command_buffer, 0);
    VkCommandBufferBeginInfo begin_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkBeginCommandBuffer(fd.command_buffer, &begin_info);

    const VkImageAspectFlags depth_barrier_aspect = getDepthBarrierAspectMask(shadow_depth_format);
    const VkDeviceSize offsets[] = { 0 };

    auto renderShadowPass = [&](const ShadowMap& sm, bool draw_geometry) {
        VkImageMemoryBarrier to_depth_attachment = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = sm.image,
            .subresourceRange = {
                .aspectMask = depth_barrier_aspect,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        vkCmdPipelineBarrier(
            fd.command_buffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &to_depth_attachment);

        VkRenderingAttachmentInfo depth_attachment = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = sm.view,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = { .depthStencil = { 1.0f, 0 } },
        };
        VkRenderingInfo rendering_info = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = { { 0, 0 }, { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE } },
            .layerCount = 1,
            .pDepthAttachment = &depth_attachment,
        };

        vkCmdBeginRendering(fd.command_buffer, &rendering_info);

        if (draw_geometry) {
            VkViewport shadow_viewport = {
                .x = 0.0f,
                .y = 0.0f,
                .width = float(SHADOW_MAP_SIZE),
                .height = float(SHADOW_MAP_SIZE),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };
            vkCmdSetViewport(fd.command_buffer, 0, 1, &shadow_viewport);

            VkRect2D shadow_scissor = { { 0, 0 }, { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE } };
            vkCmdSetScissor(fd.command_buffer, 0, 1, &shadow_scissor);

            vkCmdSetDepthBias(fd.command_buffer, depth_bias_constant, 0.0f, depth_bias_slope);

            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &sm.scene_set, 0, nullptr);
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, offsets);
            vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT16);

            auto drawShadowModel = [&](uint32_t slot, uint32_t first_index, uint32_t index_count) {
                uint32_t dynamic_offset = static_cast<uint32_t>(slot * model_ubo_stride);
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                        1, 1, &material_sets[0], 1, &dynamic_offset);
                vkCmdDrawIndexed(fd.command_buffer, index_count, 1, first_index, 0, 0);
            };

            drawShadowModel(OBJECT_SPHERE_1, 0, sphere_index_count);
            if (obj2_enabled) {
                drawShadowModel(OBJECT_SPHERE_2, 0, sphere_index_count);
            }
            if (floor_enabled) {
                drawShadowModel(OBJECT_FLOOR, floor_first_index, FLOOR_INDEX_COUNT);
            }
            if (wall_enabled) {
                drawShadowModel(OBJECT_WALL, floor_first_index, FLOOR_INDEX_COUNT);
            }
        }

        vkCmdEndRendering(fd.command_buffer);

        VkImageMemoryBarrier to_shader_read = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = sm.image,
            .subresourceRange = {
                .aspectMask = depth_barrier_aspect,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        vkCmdPipelineBarrier(
            fd.command_buffer,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &to_shader_read);
    };

    renderShadowPass(dir_shadow, dir_light_enabled && dir_shadow_enabled);
    renderShadowPass(spot_shadow, spot_shadow_enabled && !spot_lights.empty());

    VkClearValue clear_values[2] = {};
    clear_values[0].color = { { 0.1f, 0.1f, 0.15f, 1.0f } };
    clear_values[1].depthStencil = { 1.0f, 0 };
    VkRenderPassBeginInfo render_pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass, .framebuffer = fd.framebuffer,
        .renderArea = { .extent = context.swapchain_extent },
        .clearValueCount = 2, .pClearValues = clear_values,
    };
    vkCmdBeginRenderPass(fd.command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = { 0, 0, (float)context.swapchain_extent.width, (float)context.swapchain_extent.height, 0.0f, 1.0f };
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    VkRect2D scissor = { {0, 0}, context.swapchain_extent };
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &scene_set, 0, nullptr);

    const float time = static_cast<float>(anim_time);

    auto drawModel = [&](uint32_t slot, uint32_t material, const PushConstants& push,
                         uint32_t first_index, uint32_t index_count) {
        uint32_t dynamic_offset = static_cast<uint32_t>(slot * model_ubo_stride);
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                1, 1, &material_sets[material], 1, &dynamic_offset);
        vkCmdPushConstants(fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &push);
        vkCmdDrawIndexed(fd.command_buffer, index_count, 1, first_index, 0, 0);
    };

    auto objectPush = [&](uint32_t object) {
        return PushConstants{ static_cast<uint32_t>(objects[object].effect), time, objects[object].effect_strength, 0.0f };
    };
    const PushConstants no_effect = { EFFECT_NONE, time, 1.0f, 0.0f };

    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, triangle_pipeline);
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, offsets);
    vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT16);

    drawModel(OBJECT_SPHERE_1, objects[OBJECT_SPHERE_1].material, objectPush(OBJECT_SPHERE_1), 0, sphere_index_count);

    if (obj2_enabled) {
        drawModel(OBJECT_SPHERE_2, objects[OBJECT_SPHERE_2].material, objectPush(OBJECT_SPHERE_2), 0, sphere_index_count);
    }

    if (floor_enabled) {
        drawModel(OBJECT_FLOOR, objects[OBJECT_FLOOR].material, objectPush(OBJECT_FLOOR), floor_first_index, FLOOR_INDEX_COUNT);
    }

    if (wall_enabled) {
        drawModel(OBJECT_WALL, objects[OBJECT_WALL].material, objectPush(OBJECT_WALL), floor_first_index, FLOOR_INDEX_COUNT);
    }

    if (show_lights) {
        for (size_t i = 0; i < point_lights.size(); ++i) {
            drawModel(POINT_MARKER_BASE + static_cast<uint32_t>(i), 0, no_effect, 0, sphere_index_count);
        }
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            drawModel(SPOT_MARKER_BASE + static_cast<uint32_t>(i), 0, no_effect, 0, sphere_index_count);
        }

        if (!spot_lights.empty()) {
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &cone_vertex_buffer, offsets);
            for (size_t i = 0; i < spot_lights.size(); ++i) {
                uint32_t dynamic_offset = static_cast<uint32_t>((SPOT_CONE_BASE + i) * model_ubo_stride);
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                        1, 1, &material_sets[0], 1, &dynamic_offset);
                vkCmdDraw(fd.command_buffer, cone_vertex_count, 1, 0, 0);
            }
        }
    }

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application