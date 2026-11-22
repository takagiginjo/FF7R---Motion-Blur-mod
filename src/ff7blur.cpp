// Camera motion blur for Final Fantasy VII Remake (Unreal Engine 4, Direct3D 11),
// as a ReShade add-on, next to Luma.
//
// The game has Unreal's motion blur and it blurs objects only: the velocity
// buffer holds the velocity of things that move by themselves and is 0 for the
// rest, and the camera's own motion, which Unreal normally works out from the
// depth for those pixels, comes out as nothing in this game.
//
// This add-on:
//   1. Finds the velocity flatten compute shader, the first pass of the motion
//      blur, by the constants it decodes the velocity with.
//   2. Just before that shader runs, writes the camera's motion into the
//      velocity buffer wherever it is 0, worked out from the depth and the
//      ClipToPrevClip matrix in the View constants, as Unreal's temporal AA
//      does it. Things that have a velocity of their own, the player's
//      character among them, keep it.
//   3. Takes over the game's motion blur filter, the draw that reads what the
//      flatten pass wrote, and blurs with its own compute shader, built like
//      the ones in the Assassin's Creed add-ons: a shutter, a length limit,
//      near things kept sharp. If that cannot be done, the game's own blur runs
//      with the camera's motion in it.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define ImTextureID ImU64
#include <imgui.h>
#include <reshade.hpp>

#include <d3d11_1.h>
#include <wrl/client.h>

#include <blur.h>   // generated from shaders/blur.hlsl: blur_cs
#include <inject.h> // generated from shaders/inject.hlsl: inject_cs

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_set>

using Microsoft::WRL::ComPtr;
using namespace reshade::api;

extern "C" __declspec(dllexport) const char *NAME = "Camera Motion Blur (FF7 Remake)";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Adds the camera's motion to the velocity buffer, and blurs with it, camera and objects.";

static const char *SECTION = "CameraBlur";

//----------------------------------------------------------------------------
// Settings

// The shutter at 100% on both sliders, as a part of a frame's motion.
static const float TOP_SHUTTER = 0.6f;
static const int SAMPLES = 20; // taps along each blur

struct settings
{
    std::atomic<bool> enabled{true};
    std::atomic<bool> own_blur{true};  // take over the game's blur; off = the game's own, with camera motion
    std::atomic<float> strength{1.0f}; // 0..1, share of the top setting
    std::atomic<float> scale{1.0f};    // 0..1, share of the top setting; multiplies the strength
    std::atomic<int> limit{3};         // longest blur: 1, 2 or 3 % of the screen height
    std::atomic<int> near_m{3};        // things nearer than this many metres are kept sharp
    std::atomic<bool> debug{false};    // show the velocity instead of the image
};
static settings g_settings;

static void load_settings()
{
    int enabled = 1, own = 1, strength = 100, scale = 100, limit = 3, near_m = 3, debug = 0;
    reshade::get_config_value(nullptr, SECTION, "Enabled", enabled);
    reshade::get_config_value(nullptr, SECTION, "OwnBlur", own);
    reshade::get_config_value(nullptr, SECTION, "BlurAmount", strength);
    reshade::get_config_value(nullptr, SECTION, "BlurScale", scale);
    reshade::get_config_value(nullptr, SECTION, "BlurLength", limit);
    reshade::get_config_value(nullptr, SECTION, "KeepNearSharp", near_m);
    reshade::get_config_value(nullptr, SECTION, "Debug", debug);
    g_settings.enabled = enabled != 0;
    g_settings.own_blur = own != 0;
    g_settings.strength = float(std::clamp(strength, 0, 100)) / 100.0f;
    g_settings.scale = float(std::clamp(scale, 10, 100)) / 100.0f;
    g_settings.limit = std::clamp(limit, 1, 3);
    g_settings.near_m = std::clamp(near_m, 0, 10);
    g_settings.debug = debug != 0;
}

//----------------------------------------------------------------------------
// Log, next to the add-on. Small on purpose.

static FILE *g_log = nullptr;
static std::mutex g_log_mutex;
static int g_log_lines = 0;

static void mb_log(const char *fmt, ...)
{
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (!g_log || g_log_lines >= 300)
        return;
    ++g_log_lines;
    char text[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(text, sizeof(text), fmt, args);
    va_end(args);
    fprintf(g_log, "%s\n", text);
    fflush(g_log);
}

static void open_log(HMODULE module)
{
    wchar_t path[MAX_PATH];
    const DWORD n = GetModuleFileNameW(module, path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH)
        return;
    std::wstring p(path, n);
    const size_t dot = p.find_last_of(L'.');
    if (dot != std::wstring::npos)
        p.resize(dot);
    p += L".log";
    g_log = _wfopen(p.c_str(), L"w");
}

//----------------------------------------------------------------------------
// The velocity flatten compute shader, found when the game creates it.

static std::unordered_set<void *> g_flatten_cs;
static std::shared_mutex g_flatten_mutex;
static unsigned g_flatten_found = 0;

static bool is_flatten_cs(void *cs)
{
    std::shared_lock<std::shared_mutex> lock(g_flatten_mutex);
    return g_flatten_cs.count(cs) != 0;
}

static unsigned fnv1a(const void *data, size_t size)
{
    unsigned h = 2166136261u;
    const unsigned char *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i)
        h = (h ^ p[i]) * 16777619u;
    return h;
}

// True when the shader holds a 32-bit float close to `value`.
static bool has_float(const unsigned char *code, size_t size, float value)
{
    for (size_t i = 0; i + 4 <= size; ++i)
    {
        float f;
        memcpy(&f, code + i, 4);
        if (std::fabs(f - value) < 2.0e-5f)
            return true;
    }
    return false;
}

static void on_init_pipeline(device *dev, pipeline_layout, uint32_t count, const pipeline_subobject *subobjects,
                             pipeline pipe)
{
    if (dev->get_api() != device_api::d3d11 || pipe.handle == 0)
        return;
    for (uint32_t i = 0; i < count; ++i)
    {
        if (subobjects[i].type != pipeline_subobject_type::compute_shader || subobjects[i].count == 0)
            continue;
        const shader_desc *desc = static_cast<const shader_desc *>(subobjects[i].data);
        if (!desc || !desc->code || desc->code_size < 200 || desc->code_size > 20000)
            continue;
        const unsigned char *code = static_cast<const unsigned char *>(desc->code);

        // Unreal decodes its velocity with (enc - 0.499992) * 4.008016. The one
        // compute shader that does it and keeps a group-shared tile is the
        // flatten pass; its hash in this game build is AFC9976E.
        const bool by_hash = fnv1a(code, desc->code_size) == 0xAFC9976Eu;
        const bool by_constants = has_float(code, desc->code_size, 4.008016f) &&
                                  has_float(code, desc->code_size, -0.499992f);
        if (!by_hash && !by_constants)
            continue;
        {
            std::unique_lock<std::shared_mutex> lock(g_flatten_mutex);
            g_flatten_cs.insert(reinterpret_cast<void *>(pipe.handle));
            ++g_flatten_found;
        }
        mb_log("velocity flatten compute shader %p, %zu bytes (%s)", reinterpret_cast<void *>(pipe.handle),
               desc->code_size, by_hash ? "by hash" : "by its constants");
    }
}

static void on_destroy_pipeline(device *, pipeline pipe)
{
    std::unique_lock<std::shared_mutex> lock(g_flatten_mutex);
    g_flatten_cs.erase(reinterpret_cast<void *>(pipe.handle));
}

//----------------------------------------------------------------------------
// Direct3D 11 objects

static std::mutex g_state_mutex; // the add-on's own state

static ComPtr<ID3D11Device> g_dev;
static ComPtr<ID3D11ComputeShader> g_cs;
static ComPtr<ID3D11Buffer> g_cb;        // the settings
static ComPtr<ID3D11Buffer> g_view_copy; // ClipToPrevClip and the jitter, copied from the View constants
static ComPtr<ID3D11Texture2D> g_scratch;
static ComPtr<ID3D11UnorderedAccessView> g_scratch_uav;

// For the blur: what the flatten dispatch had, kept for the filter draw that
// follows, and the blur's own resources.
static ComPtr<ID3D11ShaderResourceView> g_vel_srv, g_depth_srv; // the velocity buffer and the depth
static ComPtr<ID3D11Resource> g_flat_res;     // what the flatten pass wrote, to spot the filter that reads it
static ComPtr<ID3D11Buffer> g_zconv_copy;     // View constants vector 57, 16 bytes
static ComPtr<ID3D11ComputeShader> g_blur_cs;
static ComPtr<ID3D11Buffer> g_blur_cb;
static ComPtr<ID3D11SamplerState> g_sampler;
static ComPtr<ID3D11Texture2D> g_blur_dst;
static ComPtr<ID3D11UnorderedAccessView> g_blur_uav;
static UINT g_blur_w = 0, g_blur_h = 0;
static DXGI_FORMAT g_blur_format = DXGI_FORMAT_UNKNOWN;
static bool g_blur_unsupported = false;
static bool g_filter_pending = false;
static UINT g_width = 0, g_height = 0;
static DXGI_FORMAT g_tex_format = DXGI_FORMAT_UNKNOWN;
static bool g_unsupported = false;

// The View constants' vectors 114 to 118, as bytes.
static const UINT VIEW_OFFSET = 114 * 16;
static const UINT VIEW_BYTES = 5 * 16;

struct statistics
{
    unsigned flatten_dispatches = 0, injected = 0, skipped = 0, filters_replaced = 0, filters_given_back = 0;
};
static statistics g_stats;
static unsigned g_frame = 0;
static unsigned g_reasons_logged = 0;

static void release_resources()
{
    g_vel_srv.Reset();
    g_depth_srv.Reset();
    g_flat_res.Reset();
    g_zconv_copy.Reset();
    g_blur_cs.Reset();
    g_blur_cb.Reset();
    g_sampler.Reset();
    g_blur_dst.Reset();
    g_blur_uav.Reset();
    g_blur_w = g_blur_h = 0;
    g_blur_format = DXGI_FORMAT_UNKNOWN;
    g_blur_unsupported = false;
    g_filter_pending = false;
    g_scratch_uav.Reset();
    g_scratch.Reset();
    g_cs.Reset();
    g_cb.Reset();
    g_view_copy.Reset();
    g_dev.Reset();
    g_width = g_height = 0;
    g_tex_format = DXGI_FORMAT_UNKNOWN;
    g_unsupported = false;
}

static bool ensure_resources(ID3D11Device *dev, const D3D11_TEXTURE2D_DESC &desc)
{
    if (g_unsupported)
        return false;
    // Only a change of device drops what was made; the first call notes the device.
    if (g_dev.Get() != dev)
    {
        if (g_dev)
            release_resources();
        g_dev = dev;
    }

    if (!g_cs)
    {
        if (FAILED(dev->CreateComputeShader(inject_cs, sizeof(inject_cs), nullptr, g_cs.GetAddressOf())))
        {
            mb_log("could not create the compute shader");
            g_unsupported = true;
            return false;
        }
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = 16;
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        D3D11_BUFFER_DESC cd = {};
        cd.ByteWidth = VIEW_BYTES;
        cd.Usage = D3D11_USAGE_DEFAULT;
        cd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_BUFFER_DESC zd = {};
        zd.ByteWidth = 16;
        zd.Usage = D3D11_USAGE_DEFAULT;
        zd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (FAILED(dev->CreateBuffer(&bd, nullptr, g_cb.GetAddressOf())) ||
            FAILED(dev->CreateBuffer(&cd, nullptr, g_view_copy.GetAddressOf())) ||
            FAILED(dev->CreateBuffer(&zd, nullptr, g_zconv_copy.GetAddressOf())))
        {
            mb_log("could not create the constant buffers");
            g_unsupported = true;
            return false;
        }
    }

    if (g_scratch && g_width == desc.Width && g_height == desc.Height && g_tex_format == desc.Format)
        return true;

    g_scratch_uav.Reset();
    g_scratch.Reset();

    if (desc.ArraySize != 1 || desc.MipLevels != 1 || desc.SampleDesc.Count != 1)
    {
        mb_log("the velocity texture is an array, has mips or is multisampled (%u/%u/%u), not supported",
               desc.ArraySize, desc.MipLevels, desc.SampleDesc.Count);
        g_unsupported = true;
        return false;
    }

    D3D11_FEATURE_DATA_FORMAT_SUPPORT2 fs2 = {};
    fs2.InFormat = DXGI_FORMAT_R16G16_UNORM;
    if (FAILED(dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2, &fs2, sizeof(fs2))) ||
        !(fs2.OutFormatSupport2 & D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE))
    {
        mb_log("R16G16_UNORM cannot be written by a compute shader on this GPU");
        g_unsupported = true;
        return false;
    }

    D3D11_TEXTURE2D_DESC ud = desc;
    ud.Usage = D3D11_USAGE_DEFAULT;
    ud.CPUAccessFlags = 0;
    ud.MiscFlags = 0;
    ud.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format = DXGI_FORMAT_R16G16_UNORM;
    uav.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    if (FAILED(dev->CreateTexture2D(&ud, nullptr, g_scratch.GetAddressOf())) ||
        FAILED(dev->CreateUnorderedAccessView(g_scratch.Get(), &uav, g_scratch_uav.GetAddressOf())))
    {
        mb_log("could not create the working texture (%ux%u, format %u)", desc.Width, desc.Height,
               unsigned(desc.Format));
        g_scratch_uav.Reset();
        g_scratch.Reset();
        g_unsupported = true;
        return false;
    }

    g_width = desc.Width;
    g_height = desc.Height;
    g_tex_format = desc.Format;
    mb_log("working texture ready: %ux%u, texture format %u", g_width, g_height, unsigned(g_tex_format));
    return true;
}

//----------------------------------------------------------------------------
// The injection

// Says why nothing was done, the first few times.
static void skip(const char *reason)
{
    ++g_stats.skipped;
    if (g_reasons_logged < 6)
    {
        ++g_reasons_logged;
        mb_log("nothing done at the velocity flatten pass: %s", reason);
    }
}

static void inject(ID3D11DeviceContext *ctx)
{
    ++g_stats.flatten_dispatches;

    ID3D11ShaderResourceView *raw[2] = {};
    ctx->CSGetShaderResources(0, 2, raw);
    ComPtr<ID3D11ShaderResourceView> vel_srv, depth_srv;
    vel_srv.Attach(raw[0]);
    depth_srv.Attach(raw[1]);
    if (!vel_srv || !depth_srv)
        return skip("the velocity or depth texture is not bound where expected");

    ComPtr<ID3D11Resource> res;
    vel_srv->GetResource(res.GetAddressOf());
    ComPtr<ID3D11Texture2D> vel_tex;
    if (!res || FAILED(res.As(&vel_tex)))
        return skip("the velocity texture is not a 2D texture");
    D3D11_TEXTURE2D_DESC td = {};
    vel_tex->GetDesc(&td);
    if (td.Format != DXGI_FORMAT_R16G16_UNORM && td.Format != DXGI_FORMAT_R16G16_TYPELESS)
        return skip("the velocity texture is not R16G16");

    ComPtr<ID3D11Device> dev;
    ctx->GetDevice(dev.GetAddressOf());
    if (!ensure_resources(dev.Get(), td) || !g_cs)
        return skip("the working resources could not be made");

    // The View constants, vectors 114 to 118, copied now: the copy is recorded
    // in the command stream at this dispatch.
    ComPtr<ID3D11DeviceContext1> ctx1;
    if (FAILED(ctx->QueryInterface(IID_PPV_ARGS(ctx1.GetAddressOf()))))
        return skip("no ID3D11DeviceContext1");
    ComPtr<ID3D11Buffer> view;
    UINT first = 0, num = 0;
    ctx1->CSGetConstantBuffers1(1, 1, view.GetAddressOf(), &first, &num);
    if (!view)
        return skip("the View constants are not bound at slot 1");
    D3D11_BUFFER_DESC vbd = {};
    view->GetDesc(&vbd);
    const UINT start = first * 16 + VIEW_OFFSET;
    if (start + VIEW_BYTES > vbd.ByteWidth)
        return skip("the View constants are smaller than expected");
    D3D11_BOX box = {start, 0, 0, start + VIEW_BYTES, 1, 1};
    ctx->CopySubresourceRegion(g_view_copy.Get(), 0, 0, 0, 0, view.Get(), 0, &box);

    // Vector 57 of the View constants turns a depth into a distance; the blur
    // needs it later, when the View constants may not be bound.
    const UINT zstart = first * 16 + 57 * 16;
    if (g_zconv_copy && zstart + 16 <= vbd.ByteWidth)
    {
        D3D11_BOX zbox = {zstart, 0, 0, zstart + 16, 1, 1};
        ctx->CopySubresourceRegion(g_zconv_copy.Get(), 0, 0, 0, 0, view.Get(), 0, &zbox);
    }

    // When this add-on's own blur follows, the camera's true motion goes in and
    // the blur limits it. When the game's own blur runs, the settings apply here.
    const bool own = g_settings.own_blur;
    const float constants[4] = {own ? 1.0f : g_settings.strength * g_settings.scale * 3.0f,
                                own ? 0.15f : float(g_settings.limit.load()) / 100.0f, 0.0f, 0.0f};
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    if (FAILED(ctx->Map(g_cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        return skip("could not update the constant buffer");
    memcpy(mapped.pData, constants, sizeof(constants));
    ctx->Unmap(g_cb.Get(), 0);

    // Save what the compute stage has bound; the game's own state must come back
    // untouched.
    ComPtr<ID3D11ComputeShader> old_cs;
    ctx->CSGetShader(old_cs.GetAddressOf(), nullptr, nullptr);
    ID3D11ShaderResourceView *old_srv[2] = {};
    ctx->CSGetShaderResources(0, 2, old_srv);
    ID3D11UnorderedAccessView *old_uav[2] = {};
    ctx->CSGetUnorderedAccessViews(0, 2, old_uav);
    ID3D11Buffer *old_cb[2] = {};
    ctx->CSGetConstantBuffers(0, 2, old_cb);

    ID3D11ShaderResourceView *srvs[2] = {vel_srv.Get(), depth_srv.Get()};
    ID3D11UnorderedAccessView *uavs[2] = {g_scratch_uav.Get(), nullptr};
    ID3D11Buffer *cbs[2] = {g_cb.Get(), g_view_copy.Get()};
    const UINT keep[2] = {0xFFFFFFFF, 0xFFFFFFFF};
    ctx->CSSetShader(g_cs.Get(), nullptr, 0);
    ctx->CSSetShaderResources(0, 2, srvs);
    ctx->CSSetUnorderedAccessViews(0, 2, uavs, keep);
    ctx->CSSetConstantBuffers(0, 2, cbs);
    ctx->Dispatch((td.Width + 7) / 8, (td.Height + 7) / 8, 1);

    ctx->CSSetShader(old_cs.Get(), nullptr, 0);
    ctx->CSSetShaderResources(0, 2, old_srv);
    ctx->CSSetUnorderedAccessViews(0, 2, old_uav, keep);
    ctx->CSSetConstantBuffers(0, 2, old_cb);
    for (ID3D11ShaderResourceView *s : old_srv)
        if (s)
            s->Release();
    if (old_uav[0])
    {
        ComPtr<ID3D11Resource> flat;
        old_uav[0]->GetResource(flat.GetAddressOf());
        g_flat_res = flat;
    }
    for (ID3D11UnorderedAccessView *u : old_uav)
        if (u)
            u->Release();
    for (ID3D11Buffer *b : old_cb)
        if (b)
            b->Release();

    // The velocity buffer, with the camera's motion in it, goes back; the game's
    // own dispatch, next, reads it.
    ctx->CopyResource(vel_tex.Get(), g_scratch.Get());

    // What the filter draw that follows will need.
    g_vel_srv = vel_srv;
    g_depth_srv = depth_srv;
    g_filter_pending = own;

    if (++g_stats.injected == 1)
        mb_log("first injection: velocity %ux%u, format %u, camera blur %d%%", td.Width, td.Height, unsigned(td.Format),
               int(std::lround(g_settings.strength * 100.0f)));
}

//----------------------------------------------------------------------------
// The motion blur filter, taken over

static bool give_back(const char *reason)
{
    ++g_stats.filters_given_back;
    g_filter_pending = false;
    if (g_reasons_logged < 8)
    {
        ++g_reasons_logged;
        mb_log("the game's own blur runs: %s", reason);
    }
    return false;
}

static bool ensure_blur_resources(ID3D11Device *dev, const D3D11_TEXTURE2D_DESC &desc, DXGI_FORMAT view_format)
{
    if (g_blur_unsupported)
        return false;
    if (!g_blur_cs)
    {
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = 32;
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        D3D11_SAMPLER_DESC sd = {};
        sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.MaxLOD = D3D11_FLOAT32_MAX;
        if (FAILED(dev->CreateComputeShader(blur_cs, sizeof(blur_cs), nullptr, g_blur_cs.GetAddressOf())) ||
            FAILED(dev->CreateBuffer(&bd, nullptr, g_blur_cb.GetAddressOf())) ||
            FAILED(dev->CreateSamplerState(&sd, g_sampler.GetAddressOf())))
        {
            mb_log("could not create the blur's compute shader, constant buffer or sampler");
            g_blur_unsupported = true;
            return false;
        }
    }
    if (g_blur_dst && g_blur_w == desc.Width && g_blur_h == desc.Height && g_blur_format == desc.Format)
        return true;

    g_blur_uav.Reset();
    g_blur_dst.Reset();
    if (desc.ArraySize != 1 || desc.MipLevels != 1 || desc.SampleDesc.Count != 1)
    {
        mb_log("the scene colour is an array, has mips or is multisampled, not supported");
        g_blur_unsupported = true;
        return false;
    }
    D3D11_FEATURE_DATA_FORMAT_SUPPORT2 fs2 = {};
    fs2.InFormat = view_format;
    if (FAILED(dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2, &fs2, sizeof(fs2))) ||
        !(fs2.OutFormatSupport2 & D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE))
    {
        mb_log("format %u cannot be written by a compute shader on this GPU", unsigned(view_format));
        g_blur_unsupported = true;
        return false;
    }
    D3D11_TEXTURE2D_DESC ud = desc;
    ud.Usage = D3D11_USAGE_DEFAULT;
    ud.CPUAccessFlags = 0;
    ud.MiscFlags = 0;
    ud.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format = view_format;
    uav.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    if (FAILED(dev->CreateTexture2D(&ud, nullptr, g_blur_dst.GetAddressOf())) ||
        FAILED(dev->CreateUnorderedAccessView(g_blur_dst.Get(), &uav, g_blur_uav.GetAddressOf())))
    {
        mb_log("could not create the blur's working texture");
        g_blur_uav.Reset();
        g_blur_dst.Reset();
        g_blur_unsupported = true;
        return false;
    }
    g_blur_w = desc.Width;
    g_blur_h = desc.Height;
    g_blur_format = desc.Format;
    mb_log("blur working texture ready: %ux%u, format %u", g_blur_w, g_blur_h, unsigned(desc.Format));
    return true;
}

static DXGI_FORMAT typed_format(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R11G11B10_FLOAT: return DXGI_FORMAT_R11G11B10_FLOAT;
    default: return f;
    }
}

// The draw after the flatten pass that reads what it wrote: the filter. Returns
// true when it was taken over and the game's draw is to be skipped.
static bool replace_filter(ID3D11DeviceContext *ctx)
{
    ComPtr<ID3D11RenderTargetView> rtv;
    ctx->OMGetRenderTargets(1, rtv.GetAddressOf(), nullptr);
    if (!rtv)
        return false;
    ComPtr<ID3D11Resource> rres;
    rtv->GetResource(rres.GetAddressOf());
    ComPtr<ID3D11Texture2D> tex;
    if (!rres || FAILED(rres.As(&tex)))
        return false;
    D3D11_TEXTURE2D_DESC td = {};
    tex->GetDesc(&td);
    if (td.Width < 960)
        return false; // the small tile passes read it too

    // What the draw reads: the flatten pass's output marks it as the filter.
    ID3D11ShaderResourceView *raw[16] = {};
    ctx->PSGetShaderResources(0, 16, raw);
    ComPtr<ID3D11ShaderResourceView> srvs[16];
    for (int i = 0; i < 16; ++i)
        srvs[i].Attach(raw[i]);
    bool reads_flat = false;
    int color_slot = -1;
    for (int i = 0; i < 16; ++i)
    {
        if (!srvs[i])
            continue;
        ComPtr<ID3D11Resource> r;
        srvs[i]->GetResource(r.GetAddressOf());
        if (r.Get() == g_flat_res.Get())
        {
            reads_flat = true;
            continue;
        }
        ComPtr<ID3D11Texture2D> t;
        if (color_slot < 0 && r && SUCCEEDED(r.As(&t)))
        {
            D3D11_TEXTURE2D_DESC d = {};
            t->GetDesc(&d);
            if (d.Width == td.Width && d.Height == td.Height && d.Format == td.Format && t.Get() != tex.Get())
                color_slot = i;
        }
    }
    if (!reads_flat)
        return false;

    // From here on this is the filter.
    if (g_stats.filters_replaced == 0 && g_stats.filters_given_back < 2)
    {
        mb_log("the filter draw: render target %ux%u format %u; reads:", td.Width, td.Height, unsigned(td.Format));
        for (int i = 0; i < 16; ++i)
        {
            if (!srvs[i])
                continue;
            ComPtr<ID3D11Resource> r;
            srvs[i]->GetResource(r.GetAddressOf());
            ComPtr<ID3D11Texture2D> t;
            D3D11_TEXTURE2D_DESC d = {};
            if (r && SUCCEEDED(r.As(&t)))
                t->GetDesc(&d);
            mb_log("  t%d: %ux%u format %u%s%s", i, d.Width, d.Height, unsigned(d.Format),
                   r.Get() == g_flat_res.Get() ? " (the flatten pass's output)" : "",
                   i == color_slot ? " (taken as the scene colour)" : "");
        }
    }
    if (color_slot < 0)
        return give_back("no scene colour found among what the filter reads");
    if (!g_vel_srv || !g_depth_srv || !g_zconv_copy)
        return give_back("the velocity, depth or distance constants were not kept");

    ComPtr<ID3D11Device> dev;
    ctx->GetDevice(dev.GetAddressOf());
    const DXGI_FORMAT view_format = typed_format(td.Format);
    if (!ensure_blur_resources(dev.Get(), td, view_format) || !g_blur_cs)
        return give_back("the blur's resources could not be made");

    const float shutter = g_settings.strength * g_settings.scale * TOP_SHUTTER;
    const float constants[8] = {shutter,
                                float(g_settings.limit.load()) / 100.0f,
                                float(SAMPLES),
                                float(g_frame % 64),
                                float(g_settings.near_m.load()) * 100.0f,
                                g_settings.debug ? 1.0f : 0.0f,
                                0.0f,
                                0.0f};
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    if (FAILED(ctx->Map(g_blur_cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        return give_back("could not update the blur's constant buffer");
    memcpy(mapped.pData, constants, sizeof(constants));
    ctx->Unmap(g_blur_cb.Get(), 0);

    ComPtr<ID3D11ComputeShader> old_cs;
    ctx->CSGetShader(old_cs.GetAddressOf(), nullptr, nullptr);
    ID3D11ShaderResourceView *old_srv[3] = {};
    ctx->CSGetShaderResources(0, 3, old_srv);
    ID3D11UnorderedAccessView *old_uav = nullptr;
    ctx->CSGetUnorderedAccessViews(0, 1, &old_uav);
    ID3D11Buffer *old_cb[2] = {};
    ctx->CSGetConstantBuffers(0, 2, old_cb);
    ID3D11SamplerState *old_smp = nullptr;
    ctx->CSGetSamplers(0, 1, &old_smp);

    ID3D11ShaderResourceView *in[3] = {srvs[color_slot].Get(), g_vel_srv.Get(), g_depth_srv.Get()};
    ID3D11UnorderedAccessView *uav = g_blur_uav.Get();
    ID3D11Buffer *cbs[2] = {g_blur_cb.Get(), g_zconv_copy.Get()};
    ID3D11SamplerState *smp = g_sampler.Get();
    const UINT keep = 0xFFFFFFFF;
    ctx->CSSetShader(g_blur_cs.Get(), nullptr, 0);
    ctx->CSSetShaderResources(0, 3, in);
    ctx->CSSetUnorderedAccessViews(0, 1, &uav, &keep);
    ctx->CSSetConstantBuffers(0, 2, cbs);
    ctx->CSSetSamplers(0, 1, &smp);
    ctx->Dispatch((td.Width + 7) / 8, (td.Height + 7) / 8, 1);

    ctx->CSSetShader(old_cs.Get(), nullptr, 0);
    ctx->CSSetShaderResources(0, 3, old_srv);
    ctx->CSSetUnorderedAccessViews(0, 1, &old_uav, &keep);
    ctx->CSSetConstantBuffers(0, 2, old_cb);
    ctx->CSSetSamplers(0, 1, &old_smp);
    for (ID3D11ShaderResourceView *s : old_srv)
        if (s)
            s->Release();
    if (old_uav)
        old_uav->Release();
    for (ID3D11Buffer *b : old_cb)
        if (b)
            b->Release();
    if (old_smp)
        old_smp->Release();

    // The result goes where the game's filter would have written it.
    ctx->CopyResource(tex.Get(), g_blur_dst.Get());
    g_filter_pending = false;
    if (++g_stats.filters_replaced == 1)
        mb_log("first blur ran: %ux%u, shutter %d%%, longest blur %d%%, keep sharp nearer than %d m", td.Width, td.Height,
               int(std::lround(shutter * 100.0f)), g_settings.limit.load(), g_settings.near_m.load());
    return true;
}

//----------------------------------------------------------------------------
// Events

static bool on_draw_any(command_list *cmd)
{
    if (!g_filter_pending || !g_settings.enabled || !g_settings.own_blur ||
        cmd->get_device()->get_api() != device_api::d3d11)
        return false;
    ID3D11DeviceContext *ctx = reinterpret_cast<ID3D11DeviceContext *>(cmd->get_native());
    if (!ctx)
        return false;
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (!g_filter_pending)
        return false;
    return replace_filter(ctx);
}

static bool on_draw(command_list *cmd, uint32_t, uint32_t, uint32_t, uint32_t)
{
    return on_draw_any(cmd);
}

static bool on_draw_indexed(command_list *cmd, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
{
    return on_draw_any(cmd);
}

static bool on_dispatch(command_list *cmd, uint32_t, uint32_t, uint32_t)
{
    if (!g_settings.enabled || cmd->get_device()->get_api() != device_api::d3d11)
        return false;
    ID3D11DeviceContext *ctx = reinterpret_cast<ID3D11DeviceContext *>(cmd->get_native());
    if (!ctx)
        return false;

    ComPtr<ID3D11ComputeShader> cs;
    ctx->CSGetShader(cs.GetAddressOf(), nullptr, nullptr);
    if (!cs || !is_flatten_cs(cs.Get()))
        return false;

    std::lock_guard<std::mutex> lock(g_state_mutex);
    inject(ctx);
    return false; // the game's own dispatch still runs
}

static void on_init_device(device *dev)
{
    if (dev->get_api() == device_api::d3d11)
        mb_log("Direct3D 11 device %p created", reinterpret_cast<void *>(dev->get_native()));
}

static void on_destroy_device(device *dev)
{
    // Other Direct3D 11 devices may come and go. Only our own resources, made on
    // one device, go with it.
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (dev->get_api() == device_api::d3d11 && g_dev && reinterpret_cast<void *>(dev->get_native()) == g_dev.Get())
    {
        mb_log("the device our resources were made on was destroyed");
        release_resources();
    }
}

static void on_init_effect_runtime(effect_runtime *)
{
    load_settings();
    mb_log("settings: %s, own blur %s, motion blur %d%%, scale %d%%, longest blur %d%%, keep sharp nearer than %d m",
           g_settings.enabled ? "on" : "off", g_settings.own_blur ? "on" : "off",
           int(std::lround(g_settings.strength * 100.0f)), int(std::lround(g_settings.scale * 100.0f)),
           g_settings.limit.load(), g_settings.near_m.load());
}

static void on_present(effect_runtime *)
{
    std::lock_guard<std::mutex> lock(g_state_mutex);
    ++g_frame;
    g_filter_pending = false; // nothing carries over from one frame to the next
    if (g_frame == 300 || g_frame == 1800 || g_frame % 7200 == 0)
        mb_log("frame %u: %u flatten shaders known, flatten pass dispatched %u times, camera motion added %u, "
               "nothing done %u; blur filter taken over %u times, left to the game %u",
               g_frame, g_flatten_found, g_stats.flatten_dispatches, g_stats.injected, g_stats.skipped,
               g_stats.filters_replaced, g_stats.filters_given_back);
}

//----------------------------------------------------------------------------
// The overlay window

static void draw_window(effect_runtime *)
{
    bool enabled = g_settings.enabled;
    if (ImGui::Checkbox("Enabled", &enabled))
    {
        g_settings.enabled = enabled;
        reshade::set_config_value(nullptr, SECTION, "Enabled", enabled ? 1 : 0);
    }

    bool own = g_settings.own_blur;
    if (ImGui::Checkbox("Use this add-on's blur", &own))
    {
        g_settings.own_blur = own;
        reshade::set_config_value(nullptr, SECTION, "OwnBlur", own ? 1 : 0);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("On: this add-on does the blur, camera and objects, with the settings below. "
                          "Off: the game's own blur, with the camera's motion added to it. "
                          "The game's Motion Blur option has to stay on either way.");

    int blur = int(std::lround(g_settings.strength * 100.0f));
    if (ImGui::SliderInt("Motion blur", &blur, 0, 100, "%d%%"))
    {
        g_settings.strength = float(blur) / 100.0f;
        reshade::set_config_value(nullptr, SECTION, "BlurAmount", blur);
    }
    int scale = int(std::lround(g_settings.scale * 100.0f));
    if (ImGui::SliderInt("Overall scale", &scale, 10, 100, "%d%%"))
    {
        g_settings.scale = float(scale) / 100.0f;
        reshade::set_config_value(nullptr, SECTION, "BlurScale", scale);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Multiplies the Motion blur slider.");

    int limit = std::clamp(g_settings.limit.load(), 1, 3) - 1;
    if (ImGui::Combo("Longest blur", &limit, "Low\0Medium\0High\0"))
    {
        g_settings.limit = limit + 1;
        reshade::set_config_value(nullptr, SECTION, "BlurLength", limit + 1);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The longest streak a fast camera turn can make, as a share of the screen height. "
                          "Higher settings smear more.");

    int near_m = g_settings.near_m;
    if (ImGui::SliderInt("Keep near things sharp", &near_m, 0, 10, "%d m"))
    {
        g_settings.near_m = near_m;
        reshade::set_config_value(nullptr, SECTION, "KeepNearSharp", near_m);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Things nearer than this many metres stay sharp, and do not lend their colour to the blur "
                          "of what is behind them. This is what keeps your character sharp when the camera turns. "
                          "0 blurs everything.");

    bool debug = g_settings.debug;
    if (ImGui::Checkbox("Show motion (debug)", &debug))
    {
        g_settings.debug = debug;
        reshade::set_config_value(nullptr, SECTION, "Debug", debug ? 1 : 0);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Replaces the image with the velocity: red where things move sideways, green where they "
                          "move up or down.");

    ImGui::TextDisabled("camera motion added %u times, blur taken over %u times", g_stats.injected,
                        g_stats.filters_replaced);
}

//----------------------------------------------------------------------------

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID)
{
    switch (reason)
    {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(module))
            return FALSE;
        open_log(module);
        mb_log("Camera Motion Blur (FF7 Remake) v2 loaded");
        reshade::register_event<reshade::addon_event::init_device>(on_init_device);
        reshade::register_event<reshade::addon_event::destroy_device>(on_destroy_device);
        reshade::register_event<reshade::addon_event::init_pipeline>(on_init_pipeline);
        reshade::register_event<reshade::addon_event::destroy_pipeline>(on_destroy_pipeline);
        reshade::register_event<reshade::addon_event::dispatch>(on_dispatch);
        reshade::register_event<reshade::addon_event::draw>(on_draw);
        reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
        reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
        reshade::register_event<reshade::addon_event::reshade_present>(on_present);
        reshade::register_overlay("Camera Blur", draw_window);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_addon(module);
        if (g_log)
            fclose(g_log);
        break;
    }
    return TRUE;
}
