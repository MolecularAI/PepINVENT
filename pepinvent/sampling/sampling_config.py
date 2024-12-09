from pydantic import BaseModel


class SamplingConfig(BaseModel):
    model_type: str
    model_path: str
    results_output: str
    input_sequences_path: str = ''
    batch_size: int = 1
    num_samples: int = 1000
    beam_size: int = 1000
    run_type: str = 'sampling'
