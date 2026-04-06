from argparse import ArgumentParser
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import ClassVar, List, Optional, Self


@dataclass
class Config:
    expressions: List[str]
    outDir: Path
    modelId: str
    mode: str
    templatesDir: Optional[Path]
    fromMeta: Optional[Path]
    imgSize: int
    neutralSteps: int
    neutralCfg: float
    steps: int
    cfg: float
    strength: float
    templateSeeds: int
    applySeeds: int
    baseSeed: Optional[int]
    gpuOn: bool
    numThreads: int
    dry: bool
    parallelBatchSize: int
    onlyMeta: bool

    _default_parallel_batch_size: ClassVar[int] = 5

    @classmethod
    def new(cls) -> Self:
        parser = ArgumentParser()
        parser.add_argument(
            "--model_id",
            type=str,
            default="RunDiffusion/Juggernaut-XL-v9",
            help="Model checkpoint used to generate images.",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("./suites"),
            help="Output directory for generated files (default: ./suites)",
        )
        parser.add_argument(
            "--mode",
            type=str,
            choices=["templates", "apply", "full"],
            default="full",
            help="Pipeline mode: templates (neutral only), apply (AUs from templates), full (templates + apply).",
        )
        parser.add_argument(
            "--templates_dir",
            type=Path,
            help="Template root directory for apply mode. Expected to contain ./meta and ./images.",
        )
        parser.add_argument(
            "--from_meta",
            type=Path,
            help="Render directly from existing meta json file or directory.",
        )
        parser.add_argument("--image_size", type=int, default=1024)
        parser.add_argument(
            "--neutral_steps",
            type=int,
            default=20,
            help="Denoising steps for neutral template generation.",
        )
        parser.add_argument(
            "--neutral_cfg",
            type=float,
            default=3.0,
            help="CFG for neutral template generation.",
        )
        parser.add_argument(
            "--steps",
            type=int,
            default=24,
            help="Denoising steps for AU application pass.",
        )
        parser.add_argument(
            "--cfg",
            type=float,
            default=5.0,
            help="CFG for AU application pass.",
        )
        parser.add_argument(
            "--strength",
            type=float,
            default=0.35,
            help="img2img strength for AU application pass (0.0-1.0).",
        )
        parser.add_argument(
            "--seeds",
            type=int,
            default=1,
            help="Legacy alias for --template_seeds.",
        )
        parser.add_argument(
            "--template_seeds",
            type=int,
            help="Neutral templates per demographic combination.",
        )
        parser.add_argument(
            "--apply_seeds",
            type=int,
            default=1,
            help="AU output variants per template/expression pair.",
        )
        parser.add_argument(
            "--base_seed",
            type=int,
            help="Optional base seed for reproducible random seed generation.",
        )
        parser.add_argument(
            "--gpu_on",
            default=False,
            action="store_true",
            help="Use CUDA GPU if available.",
        )
        parser.add_argument("--num_threads", type=int, default=min(8, cpu_count() or 8))
        parser.add_argument("--expressions", nargs="+", type=str, default=[])
        parser.add_argument(
            "--parallel_batch_size",
            type=int,
            default=Config._default_parallel_batch_size,
            help="How many render tasks run in parallel per CPU batch.",
        )
        parser.add_argument(
            "--dry",
            default=False,
            action="store_true",
            help="Dry run: print config and estimated file counts.",
        )
        parser.add_argument(
            "--only_meta",
            default=False,
            action="store_true",
            help="Only generate metadata json files, skip image rendering.",
        )

        args = parser.parse_args()
        template_seeds = args.template_seeds if args.template_seeds is not None else args.seeds

        if template_seeds < 1:
            raise ValueError("--template_seeds/--seeds must be >= 1")
        if args.apply_seeds < 1:
            raise ValueError("--apply_seeds must be >= 1")
        if not 0.0 <= float(args.strength) <= 1.0:
            raise ValueError("--strength must be in [0.0, 1.0]")

        return cls(
            expressions=args.expressions,
            outDir=args.output,
            modelId=args.model_id,
            mode=args.mode,
            templatesDir=args.templates_dir,
            fromMeta=args.from_meta,
            imgSize=args.image_size,
            neutralSteps=max(1, int(args.neutral_steps)),
            neutralCfg=float(args.neutral_cfg),
            steps=max(1, int(args.steps)),
            cfg=float(args.cfg),
            strength=float(args.strength),
            templateSeeds=int(template_seeds),
            applySeeds=int(args.apply_seeds),
            baseSeed=args.base_seed,
            gpuOn=args.gpu_on,
            numThreads=args.num_threads,
            dry=args.dry,
            parallelBatchSize=args.parallel_batch_size,
            onlyMeta=args.only_meta,
        )
