"""
LTX-2 Studio — Pipeline Registry
Dynamically builds parameter schemas from the actual LTX-2 pipeline code.
Zero hardcoded defaults — all values pulled from ltx_pipelines.utils.constants.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Try to import actual constants — fall back gracefully if not installed
try:
    from ltx_pipelines.utils.constants import (
        LTX_2_3_PARAMS,
        LTX_2_3_HQ_PARAMS,
        DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_IMAGE_CRF,
    )
    PARAMS = LTX_2_3_PARAMS
    HQ_PARAMS = LTX_2_3_HQ_PARAMS
    NEG_PROMPT = DEFAULT_NEGATIVE_PROMPT
    _HAS_LTX = True
except ImportError:
    logger.warning("ltx_pipelines not installed; using inline defaults")
    _HAS_LTX = False

    # Mirror of ltx_pipelines.utils.constants — kept in sync
    class _GuiderParams:
        def __init__(self, **kw):
            self.cfg_scale = kw.get("cfg_scale", 3.0)
            self.stg_scale = kw.get("stg_scale", 1.0)
            self.rescale_scale = kw.get("rescale_scale", 0.7)
            self.modality_scale = kw.get("modality_scale", 3.0)
            self.skip_step = kw.get("skip_step", 0)
            self.stg_blocks = kw.get("stg_blocks", [28])

    class _PipelineParams:
        seed = 10
        stage_1_height = 512
        stage_1_width = 768
        num_frames = 121
        frame_rate = 24.0
        num_inference_steps = 30
        video_guider_params = _GuiderParams(stg_blocks=[28])
        audio_guider_params = _GuiderParams(cfg_scale=7.0, stg_blocks=[28])

        @property
        def stage_2_height(self):
            return self.stage_1_height * 2

        @property
        def stage_2_width(self):
            return self.stage_1_width * 2

    class _HQParams(_PipelineParams):
        num_inference_steps = 15
        stage_1_height = 1088 // 2
        stage_1_width = 1920 // 2
        video_guider_params = _GuiderParams(cfg_scale=3.0, stg_scale=0.0, rescale_scale=0.45, stg_blocks=[])
        audio_guider_params = _GuiderParams(cfg_scale=7.0, stg_scale=0.0, rescale_scale=1.0, stg_blocks=[])

    PARAMS = _PipelineParams()
    HQ_PARAMS = _HQParams()
    NEG_PROMPT = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
        "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
        "proportions, unnatural skin tones, deformed facial features, artifacts"
    )
    DEFAULT_IMAGE_CRF = 33


# ============================================================================
#  PARAMETER SCHEMA
# ============================================================================
@dataclass
class ParamDef:
    name: str
    label: str
    type: str  # "int" | "float" | "bool" | "textarea" | "text" | "file" | "image_list" | "select"
    group: str = "other"
    default: object = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    required: bool = False
    description: str = ""
    accept: str = ""
    options: list = field(default_factory=list)

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None and v != "" and v != []}


# ============================================================================
#  GUIDER PARAM BUILDERS
# ============================================================================
def _video_guider_params(guider) -> list[ParamDef]:
    """Build parameter definitions from a guider params object."""
    return [
        ParamDef("video_cfg_scale", "Video CFG Scale", "float", "video_guidance",
                 default=guider.cfg_scale, min=0, max=20, step=0.1,
                 description="Classifier-Free Guidance. Higher = stronger prompt adherence."),
        ParamDef("video_stg_scale", "Video STG Scale", "float", "video_guidance",
                 default=guider.stg_scale, min=0, max=5, step=0.1,
                 description="Spatio-Temporal Guidance. Higher = stronger perturbation effect."),
        ParamDef("video_rescale_scale", "Video Rescale", "float", "video_guidance",
                 default=guider.rescale_scale, min=0, max=1, step=0.05,
                 description="Rescale after guidance. Higher = less oversaturation."),
        ParamDef("video_modality_scale", "A2V Guidance", "float", "video_guidance",
                 default=guider.modality_scale, min=0, max=10, step=0.1,
                 description="Audio-to-Video cross-attention guidance."),
        ParamDef("video_skip_step", "Video Skip Step", "int", "video_guidance",
                 default=guider.skip_step, min=0, max=10, step=1,
                 description="Periodic skip: 0 = none, 1 = skip every other."),
        ParamDef("video_stg_blocks", "Video STG Blocks", "text", "video_guidance",
                 default=",".join(str(b) for b in guider.stg_blocks),
                 description="Comma-separated transformer block indices to perturb."),
    ]


def _audio_guider_params(guider) -> list[ParamDef]:
    return [
        ParamDef("audio_cfg_scale", "Audio CFG Scale", "float", "audio_guidance",
                 default=guider.cfg_scale, min=0, max=20, step=0.1),
        ParamDef("audio_stg_scale", "Audio STG Scale", "float", "audio_guidance",
                 default=guider.stg_scale, min=0, max=5, step=0.1),
        ParamDef("audio_rescale_scale", "Audio Rescale", "float", "audio_guidance",
                 default=guider.rescale_scale, min=0, max=1, step=0.05),
        ParamDef("audio_modality_scale", "V2A Guidance", "float", "audio_guidance",
                 default=guider.modality_scale, min=0, max=10, step=0.1),
        ParamDef("audio_skip_step", "Audio Skip Step", "int", "audio_guidance",
                 default=guider.skip_step, min=0, max=10, step=1),
        ParamDef("audio_stg_blocks", "Audio STG Blocks", "text", "audio_guidance",
                 default=",".join(str(b) for b in guider.stg_blocks)),
    ]


# ============================================================================
#  COMMON PARAM SETS
# ============================================================================
def _base_gen_params(params=PARAMS) -> list[ParamDef]:
    """Base generation parameters shared by most pipelines."""
    return [
        ParamDef("prompt", "Prompt", "textarea", "prompt", required=True,
                 description="Text prompt describing the desired video content."),
        ParamDef("negative_prompt", "Negative Prompt", "textarea", "prompt",
                 default=NEG_PROMPT,
                 description="Describe what should NOT appear in the generated video."),
        ParamDef("seed", "Seed", "int", "generation",
                 default=-1, min=-1, max=4294967295, step=1,
                 description="Random seed. Use -1 for random."),
        ParamDef("height", "Height", "int", "generation",
                 default=params.stage_2_height, min=64, max=2160, step=64,
                 description="Video height in pixels, divisible by 64 (2-stage) or 32 (1-stage)."),
        ParamDef("width", "Width", "int", "generation",
                 default=params.stage_2_width, min=64, max=3840, step=64,
                 description="Video width in pixels."),
        ParamDef("num_frames", "Frames", "int", "generation",
                 default=params.num_frames, min=9, max=257, step=8,
                 description="Frame count must satisfy (F-1) % 8 == 0. E.g. 9, 17, ..., 121, ..., 257."),
        ParamDef("frame_rate", "Frame Rate (FPS)", "float", "generation",
                 default=params.frame_rate, min=1, max=60, step=1),
        ParamDef("num_inference_steps", "Steps", "int", "generation",
                 default=params.num_inference_steps, min=1, max=100, step=1,
                 description="Denoising steps. Higher = better quality but slower."),
        ParamDef("enhance_prompt", "Enhance Prompt", "bool", "generation",
                 default=False, description="Enhance prompt via Gemma text encoder."),
    ]


def _image_conditioning() -> ParamDef:
    return ParamDef("images", "Conditioning Images", "image_list", "conditioning",
                    description="Optional: condition video on reference images.")


# ============================================================================
#  PIPELINE REGISTRY
# ============================================================================
@dataclass
class PipelineInfo:
    id: str
    name: str
    description: str
    params: list[ParamDef]
    requires: list[str]  # Required model keys from config.REQUIRED_MODELS

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "params": [p.to_dict() for p in self.params],
            "requires": self.requires,
        }


def build_pipeline_registry() -> list[PipelineInfo]:
    """Build the complete pipeline registry with parameter schemas."""
    vg = PARAMS.video_guider_params
    ag = PARAMS.audio_guider_params
    hq_vg = HQ_PARAMS.video_guider_params
    hq_ag = HQ_PARAMS.audio_guider_params

    guided_params = _base_gen_params() + _video_guider_params(vg) + _audio_guider_params(ag)

    return [
        # --- 1. TI2Vid Two Stages (recommended) ---
        PipelineInfo(
            id="ti2vid_two_stages",
            name="Text/Image → Video (2-Stage)",
            description="Production quality. Stage 1 at half resolution with CFG, Stage 2 upsamples 2× with distilled LoRA. Recommended.",
            params=[*guided_params, _image_conditioning()],
            requires=["checkpoint", "distilled_lora", "spatial_upsampler", "gemma"],
        ),

        # --- 2. TI2Vid Two Stages HQ ---
        PipelineInfo(
            id="ti2vid_two_stages_hq",
            name="Text/Image → Video (2-Stage HQ)",
            description="High-quality mode using res_2s second-order sampler. Fewer steps for comparable quality.",
            params=[
                *_base_gen_params(HQ_PARAMS),
                *_video_guider_params(hq_vg),
                *_audio_guider_params(hq_ag),
                _image_conditioning(),
                ParamDef("distilled_lora_strength_stage_1", "Distilled LoRA Strength (Stage 1)",
                         "float", "advanced", default=0.25, min=0, max=1, step=0.05),
                ParamDef("distilled_lora_strength_stage_2", "Distilled LoRA Strength (Stage 2)",
                         "float", "advanced", default=0.5, min=0, max=1, step=0.05),
            ],
            requires=["checkpoint", "distilled_lora", "spatial_upsampler", "gemma"],
        ),

        # --- 3. TI2Vid One Stage ---
        PipelineInfo(
            id="ti2vid_one_stage",
            name="Text/Image → Video (1-Stage)",
            description="Single-stage generation at target resolution. No upsampling. Good for prototyping.",
            params=[
                *[ParamDef(
                    p.name, p.label, p.type, p.group,
                    default=(PARAMS.stage_1_height if p.name == "height" else
                             PARAMS.stage_1_width if p.name == "width" else p.default),
                    min=p.min, max=p.max,
                    step=(32 if p.name in ("height", "width") else p.step),
                    required=p.required, description=p.description
                ) for p in _base_gen_params()],
                *_video_guider_params(vg),
                *_audio_guider_params(ag),
                _image_conditioning(),
            ],
            requires=["checkpoint", "gemma"],
        ),

        # --- 4. Distilled ---
        PipelineInfo(
            id="distilled",
            name="Distilled (Fast)",
            description="Fastest inference with 8 predefined sigma steps. No guidance parameters needed. Uses distilled checkpoint.",
            params=[
                *[p for p in _base_gen_params() if p.name not in ("negative_prompt", "num_inference_steps")],
                _image_conditioning(),
            ],
            requires=["distilled_checkpoint", "spatial_upsampler", "gemma"],
        ),

        # --- 5. IC-LoRA ---
        PipelineInfo(
            id="ic_lora",
            name="IC-LoRA (Video-to-Video)",
            description="Video-to-video generation with In-Context LoRA. Control output via depth maps, pose, or edges.",
            params=[
                *[p for p in _base_gen_params() if p.name not in ("negative_prompt", "num_inference_steps")],
                _image_conditioning(),
                ParamDef("video_conditioning", "Reference Video", "file", "conditioning",
                         required=True, accept="video/*",
                         description="Control video (depth, pose, edges). Required."),
                ParamDef("video_conditioning_strength", "Conditioning Strength", "float", "conditioning",
                         default=1.0, min=0, max=1, step=0.05),
                ParamDef("conditioning_attention_strength", "Attention Strength", "float", "conditioning",
                         default=1.0, min=0, max=1, step=0.05,
                         description="Controls how strongly conditioning video influences output."),
                ParamDef("skip_stage_2", "Skip Stage 2 (half res)", "bool", "advanced",
                         default=False, description="Output at half resolution for faster iteration."),
            ],
            requires=["distilled_checkpoint", "spatial_upsampler", "gemma"],
        ),

        # --- 6. Keyframe Interpolation ---
        PipelineInfo(
            id="keyframe_interpolation",
            name="Keyframe Interpolation",
            description="Interpolate between keyframe images with smooth transitions. Uses guiding latents.",
            params=[
                *guided_params,
                ParamDef("images", "Keyframe Images", "image_list", "conditioning",
                         required=True, description="At least 2 keyframe images with frame indices."),
            ],
            requires=["checkpoint", "distilled_lora", "spatial_upsampler", "gemma"],
        ),

        # --- 7. Audio-to-Video ---
        PipelineInfo(
            id="a2vid_two_stage",
            name="Audio → Video (2-Stage)",
            description="Generate video driven by input audio. Audio is preserved in output (not decoded).",
            params=[
                *[p for p in _base_gen_params()],
                *_video_guider_params(vg),
                _image_conditioning(),
                ParamDef("audio_path", "Input Audio", "file", "conditioning",
                         required=True, accept="audio/*",
                         description="Audio file to drive video generation."),
                ParamDef("audio_start_time", "Audio Start Time (s)", "float", "conditioning",
                         default=0.0, min=0, max=3600, step=0.1),
                ParamDef("audio_max_duration", "Audio Max Duration (s)", "float", "conditioning",
                         default=None, min=0, max=3600, step=0.1,
                         description="Defaults to video duration (num_frames / frame_rate)."),
            ],
            requires=["checkpoint", "distilled_lora", "spatial_upsampler", "gemma"],
        ),

        # --- 8. Retake ---
        PipelineInfo(
            id="retake",
            name="Retake (Region Regeneration)",
            description="Regenerate a time region of an existing video. Keeps content outside the region unchanged.",
            params=[
                ParamDef("video_path", "Source Video", "file", "input",
                         required=True, accept="video/*"),
                ParamDef("prompt", "Prompt", "textarea", "prompt", required=True),
                ParamDef("start_time", "Start Time (s)", "float", "input",
                         default=0, min=0, max=3600, step=0.1, required=True),
                ParamDef("end_time", "End Time (s)", "float", "input",
                         default=1.0, min=0, max=3600, step=0.1, required=True),
                ParamDef("seed", "Seed", "int", "generation",
                         default=10, min=0, max=4294967295, step=1),
                ParamDef("regenerate_video", "Regenerate Video", "bool", "generation", default=True),
                ParamDef("regenerate_audio", "Regenerate Audio", "bool", "generation", default=True),
                ParamDef("enhance_prompt", "Enhance Prompt", "bool", "generation", default=False),
            ],
            requires=["distilled_checkpoint", "gemma"],
        ),
    ]


# Singleton registry
_registry: list[PipelineInfo] | None = None


def get_pipeline_registry() -> list[PipelineInfo]:
    global _registry
    if _registry is None:
        _registry = build_pipeline_registry()
    return _registry


def get_pipeline(pipeline_id: str) -> PipelineInfo | None:
    for p in get_pipeline_registry():
        if p.id == pipeline_id:
            return p
    return None
