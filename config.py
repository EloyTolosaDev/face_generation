from dataclasses import dataclass
from typing import List, Optional, Self, ClassVar
from pathlib import Path
from argparse import ArgumentParser
from os import cpu_count

@dataclass
class Config:
    expressions: List[str]
    outDir: Path
    modelId: str
    imgSize: int
    steps: int
    cfg: float
    seeds: int
    gpuOn: bool
    numThreads: int
    dry: bool
    parallelBatchSize: int
    onlyMeta: bool
    fromMeta: Optional[Path]
    
    ## default values
    _default_parallel_batch_size: ClassVar[int] = 5

    @classmethod
    def new(cls) -> Self:
        """Starts a new argument parser looking for the parameters set in Config, parses them
        and creates a new Config object with the passed parameters"""

        p = ArgumentParser()
        p.add_argument(
            "--model_id",
            type=str,
            default="RunDiffusion/Juggernaut-XL-v9",
            help="Model checkpoint to generate images from.",
        )
        p.add_argument(
            "--output",
            type=Path,
            help="Output directory for generated files (for example: ./suites)",
        )
        p.add_argument("--image_size", type=int, default=1024)
        p.add_argument("--steps", type=int, default=20)
        p.add_argument("--cfg", type=float, default=3.0)
        p.add_argument(
            "--seeds",
            type=int,
            default=1,
            help="Amount of images per combination of age/ethnicity/gender/expression",
        )
        p.add_argument(
            "--gpu_on",
            default=False,
            action="store_true",
            help="Use CUDA GPU if available",
        )
        p.add_argument("--num_threads", type=int, default=min(8, cpu_count() or 8))
        p.add_argument("--expressions", nargs="+", type=str, default=[])
        p.add_argument(
            "--parallel_batch_size",
            type=int,
            default=Config._default_parallel_batch_size,
            help="How many create_image(meta) calls run in parallel per batch",
        )
        p.add_argument(
            "--dry",
            default=False,
            action="store_true",
            help="Dry run: print config and expected file count",
        )
        p.add_argument(
            "--only_meta",
            default=False,
            action="store_true",
            help="Only meta means only generate json files needed to generate the images. By default, meta AND images are generated"
        )
        p.add_argument("--from_meta", type=Path, help="Only generate images from meta-files from a path where meta-files are on")

        args = p.parse_args()
        return cls(
            expressions=args.expressions,
            outDir=args.output,
            modelId=args.model_id,
            imgSize=args.image_size,
            steps=args.steps,
            cfg=args.cfg,
            seeds=args.seeds,
            gpuOn=args.gpu_on,
            numThreads=args.num_threads,
            dry=args.dry,
            parallelBatchSize=args.parallel_batch_size,
            onlyMeta=args.only_meta,
            fromMeta=args.from_meta,
        )
