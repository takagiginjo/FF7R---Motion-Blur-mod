// Motion blur for Final Fantasy VII Remake (Unreal Engine 4, Direct3D 11).
//
// It stands in for the game's own motion blur filter, and is built the way the
// ones in the Assassin's Creed add-ons are, so the same settings mean the same
// thing. It reads the scene colour (t0), the velocity buffer (t1, R16G16_UNORM,
// with the camera's motion added by the other shader of this add-on) and the
// depth (t2):
//
//  - Each pixel is averaged along its own velocity, from half a blur behind to
//    half ahead, as if the shutter had stayed open for `shutter` of a frame,
//    with taps spread by noise.
//  - The blur is never longer than `max_length` of the screen height.
//  - Things near the camera are kept sharp, and do not lend their colour to the
//    blur of things behind them: the player's character, when the camera turns
//    round it, is the reason. The distance comes from the depth the way
//    Unreal's shaders do it, with View.InvDeviceZToWorldZTransform (b1, vector
//    57 of the View constants): d * x + y + 1 / (d * z - w), in centimetres.
//
// The velocity is decoded the way Unreal's shaders do: (enc - 0.499992) *
// 4.008016, in clip space (2 across the screen, y up), and is the current
// position minus the previous one.

cbuffer constants : register(b0)
{
    float shutter;    // the part of a frame's motion that is blurred
    float max_length; // longest blur, as a fraction of the screen height
    float samples;    // taps along the blur
    float frame;      // changes every frame, moves the noise
    float near_cm;    // distance below which things are kept sharp
    float debug;      // 1 = show the velocity instead of the image
    float padding0;
    float padding1;
};

cbuffer depth_params : register(b1)
{
    float4 inv_z; // View.InvDeviceZToWorldZTransform
};

Texture2D<float4> color_tex : register(t0);
Texture2D<float4> vel_tex : register(t1);
Texture2D<float4> depth_tex : register(t2);
SamplerState linear_clamp : register(s0);
RWTexture2D<float4> target : register(u0);

float smooth01(float x)
{
    x = saturate(x);
    return x * x * (3.0 - 2.0 * x);
}

// Distance from the camera, in centimetres, from the depth at a UV.
float distance_at(float2 uv)
{
    uint dw, dh;
    depth_tex.GetDimensions(dw, dh);
    int2 p = clamp(int2(uv * float2(dw, dh)), int2(0, 0), int2(int(dw) - 1, int(dh) - 1));
    float d = depth_tex.Load(int3(p, 0)).x;
    return abs(d * inv_z.x + inv_z.y + 1.0 / (d * inv_z.z - inv_z.w));
}

// Interleaved gradient noise: spreads the taps so a few of them don't band.
float noise(uint2 p, float f)
{
    return frac(52.9829189 * frac(dot(float2(p) + f * 5.588238, float2(0.06711056, 0.00583715))));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    target.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
        return;

    float2 size = float2(width, height);
    float2 uv = (float2(id.xy) + 0.5) / size;

    uint vw, vh;
    vel_tex.GetDimensions(vw, vh);
    int2 vp = int2(min(uint2(uv * float2(vw, vh)), uint2(vw - 1, vh - 1)));
    float2 enc = vel_tex.Load(int3(vp, 0)).xy;
    float2 v_clip = (enc - 0.499992) * 4.008016;
    // Clip space to UV: half as long, and y runs the other way.
    float2 mv = float2(v_clip.x * 0.5, v_clip.y * -0.5);

    float4 here = color_tex.SampleLevel(linear_clamp, uv, 0.0);

    if (debug > 0.5)
    {
        // Red and green: how far the pixel moved sideways and up or down.
        target[id.xy] = float4(saturate(abs(mv) * 20.0), 0.0, 1.0);
        return;
    }

    // The blur's length: the velocity, the shutter, and the fade that keeps
    // near things sharp.
    float fade = smooth01((distance_at(uv) - near_cm) / (0.4 * near_cm + 1.0));
    float2 v = mv * shutter * fade;

    float aspect = size.x / size.y;
    float len = length(v * float2(aspect, 1.0));
    if (len > max_length)
    {
        v *= max_length / len;
        len = max_length;
    }

    // Less than a pixel of motion: nothing to blur.
    if (len * size.y < 1.0)
    {
        target[id.xy] = here;
        return;
    }

    int n = clamp(int(samples), 2, 32);
    float jitter = noise(id.xy, frame) - 0.5;
    float3 sum = float3(0.0, 0.0, 0.0);
    float weight = 0.0;
    [loop]
    for (int i = 0; i < n; ++i)
    {
        // From half a blur behind to half ahead, each tap moved a little by the
        // noise inside its own slot.
        float t = (float(i) + 0.5 + jitter) / float(n) - 0.5;
        float2 tap_uv = uv + v * t;
        // A near tap does not count: it would drag the near thing into the
        // blur of what is behind it.
        float w = smooth01((distance_at(tap_uv) - near_cm) / (3.0 * near_cm + 1.0));
        sum += color_tex.SampleLevel(linear_clamp, tap_uv, 0.0).rgb * w;
        weight += w;
    }
    // Every tap near: nothing to blur with.
    float3 color = weight > 0.0 ? sum / weight : here.rgb;
    target[id.xy] = float4(color, here.a);
}
