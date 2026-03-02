from pydantic import BaseModel

type Example = dict[str, Any]
type Examples = dict[str, list[Any]]
type Tokenizer = Any
type Model = Any


class DataConfig(BaseModel):
    dataset_type: str = "jsonl_chat" # jsonl_chat, raw
    path: str = "dataset/train_dataset.jsonl"
    chunk_size: int = 512 # 512 characters per utterance

class ModelConfig(BaseModel):
    model_path: str = "Qwen/Qwen1.5-1.8B"
    model_kwargs: dict | None = None
    finetune_mode: str = "sft" # sft or lora
    lora_kwargs: dict | None = None 

class TrainConfig(BaseModel):
    seed: int = 1234
    log_path: str 
    output_dir: str
    
    # data
    batch_size: int = 64
    num_train_epochs: int = 1
    dataloader_prefetch_factor: int = 2
    dataloader_num_workers: int = 4
    
    # optimizer
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    learning_rate: float = 1e-5
    weight_decay: float = 0
    
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_ratio: float = 0.1

    # distributed training
    deepspeed: str = "conf/ds_zero2.json" # ds_zero3 if not enough memory
    gradient_checkpointing: bool = True # true if not enough memory
    ddp_backend: str = "gloo" # for nvidia, can use nccl, otherwise use gloo

    # save
    save_strategy: str = "steps"
    save_steps: int = 250
    save_total_limit: int | None = None

    # logging
    logging_strategy: str = "steps"
    log_steps: int = 50


class ConfigV1(BaseModel):
    version: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

class Config(BaseModel):
    version: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig