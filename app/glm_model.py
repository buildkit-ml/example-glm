import sys
from typing import Dict

import torch.cuda

sys.path.append("./")
from common import FastInferenceInterface
from glm_utils import *


class GLMModel(FastInferenceInterface):
    def __init__(self, model_name: str, args) -> None:
        assert (torch.cuda.is_available())
        super().__init__(model_name, args)
        # self.device = torch.device('cuda', args.cuda_id)
        self.device = torch.cuda.current_device()
        self.batch_size = args.batch_size
        self.model_name = model_name
        self.upload_token = args.upload_token
        try:
            self.model, self.tokenizer = initialize_model_and_tokenizer(args)
            self.end_tokens = [self.tokenizer.get_command("eop"), self.tokenizer.get_command("eos")]
        except Exception as e:
            print('<GLMModel>__init__: Exception in model initialization inference:', e)
            error = traceback.format_exc()
            print(error)
            raise e
        if args.quantization_bit_width is not None:
            assert self.model_name.endswith(str(args.quantization_bit_width))
        print(f"<GLMModel>__init__: rank-{dist.get_rank()} finished!")

    def infer(self, job_ids=None, args=None) -> Dict:
        coord_url = os.environ.get("COORDINATOR_URL", "localhost:8093/my_coord")
        worker_name = os.environ.get("WORKER_NAME", "planetv2")
        try:
            raw_text = ""
            config = {}
            if dist.get_rank() == 0:
                assert isinstance(job_ids, list)
                for job_id in job_ids:
                    res = requests.patch(
                        f"http://{coord_url}/api/v1/g/jobs/atomic_job/{job_id}",
                        json={
                            "status": "running",
                        }
                    ).json()
                    print(f"Job <{job_id}> {res['status']}")
                    print(f"Job <{job_id}> has been batched.")

                raw_text = args['prompt']
                config = {
                    'seed': args['seed'],
                    'temperature': args['temperature'],
                    'top_k': args['top_k'],
                    'top_p': args['top_p'],
                    'max_tokens': max(args['max_tokens'], 1),
                    'prompt_embedding': args['prompt_embedding'],
                    'logprobs': args['logprobs']
                }

            torch.cuda.empty_cache()
            dist.barrier()
            if dist.get_rank() == 0:
                dist.broadcast_object_list([raw_text, config])
            else:
                info = [raw_text, config]
                dist.broadcast_object_list(info)
                raw_text, config = info
            dist.barrier()

            print(f"Rank-<{dist.get_rank()}> join inference.")
            start_time = time.time()
            batch_size = min(len(raw_text), self.batch_size)
            num_iter = math.ceil(len(raw_text) / batch_size)
            answers = []
            last_layer_embedding = []
            top_logprobs = []
            if config['seed'] is not None:
                torch.manual_seed(config['seed'])
                np.random.seed(config['seed'])
                random.seed(config['seed'])
                # if debug_print:
                print(f"<Main> Rank-<{dist.get_rank()}> setup random seed: {config['seed']}")

            for iter_i in range(num_iter):
                current_raw_text = raw_text[iter_i * batch_size: (iter_i + 1) * batch_size]
                if config['temperature'] == 0:
                    strategy = BaseStrategy(batch_size=len(current_raw_text), temperature=1, top_k=1,
                                            top_p=config['top_p'], end_tokens=self.end_tokens)
                else:
                    strategy = BaseStrategy(batch_size=len(current_raw_text), temperature=config['temperature'],
                                            top_k=config['top_k'], top_p=config['top_p'], end_tokens=self.end_tokens)

                cur_answer, cur_last_layer_embedding, cur_top_logprobs = fill_blanks_efficient(
                    current_raw_text, self.model, self.tokenizer, strategy, config)
                answers.extend(cur_answer)
                if cur_last_layer_embedding is None:
                    last_layer_embedding = None
                else:
                    last_layer_embedding.extend(cur_last_layer_embedding)
                if cur_top_logprobs is None:
                    top_logprobs = None
                else:
                    top_logprobs.extend(cur_top_logprobs)
                if dist.get_rank() == 0:
                    print(f"<Main> Current iter handled: {len(answers)}/{len(raw_text)}")
            end_time = time.time()
            print(f"<GLMModel>__inf__: current batch runtime: {end_time - start_time}.")
            if dist.get_rank() == 0:
                prompt_str_lengths = []
                for text in raw_text:
                    prompt_str_lengths.append(len(text))
                result = to_result(answers, args, prompt_str_lengths, last_layer_embedding, top_logprobs,
                                   job_ids=job_ids, working_directory=args.working_directory,
                                   upload_token=self.upload_token)
                for i in range(len(job_ids)):
                    job_id = job_ids[i]
                    return_payload = {
                        'request': args,
                        'result': result[i],
                    }
                    requests.patch(
                        f"http://{coord_url}/api/v1/g/jobs/atomic_job/{job_id}",
                        json={
                            "status": "finished",
                            "output": return_payload,
                            "processed_by": worker_name,
                        },
                    )
            return {"worker_states": "finished"}
        except Exception as e:
            error = traceback.format_exc()
            if dist.get_rank() == 0:
                for i in range(len(job_ids)):
                    job_id = job_ids[i]
                    requests.patch(
                        f"http://{coord_url}/api/v1/g/jobs/atomic_job/{job_id}",
                        json={
                            "status": "failed",
                            "message": error,
                            "processed_by": worker_name,
                        },
                    )
            print(error)
            raise e
