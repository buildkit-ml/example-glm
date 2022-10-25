import asyncio
import os
from typing import Dict
import json
import nats
import traceback
import torch.distributed as dist


class FastInferenceInterface:
    def __init__(self, model_name: str, args=None) -> None:
        self.model_name = model_name
        self.model_id = args.together_model_id
    def infer(self, job_id, args) -> Dict:
        pass

    async def on_message_prime_worker(self, msg):
        nats_url = os.environ.get("NATS_URL", "localhost:8092/my_coord")
        nc = await nats.connect(f"nats://{nats_url}")
        await nc.publish(self.model_name+f".{self.model_id}", b"START")
        instruction = json.loads(msg.data.decode("utf-8"))
        instruction['args'] = json.loads(instruction['args'])
        try:
            if isinstance(instruction['prompt'], list):
                instruction['args']['prompt'] = instruction['prompt']
            elif isinstance(instruction['prompt'], str):
                instruction['args']['prompt'] = [instruction['prompt']]
            else:
                raise TypeError("Only str or list of str are allowed")
        except Exception as e:
            traceback.print_exc()
            print("error in inference: "+str(e))

        if 'temperature' not in instruction['args']:
            instruction['args']['temperature'] = 0.9
        if 'top_p'not in instruction['args']:
            instruction['args']['top_p'] = 0
        if 'top_k'not in instruction['args']:
            instruction['args']['top_k'] = 50
        if 'max_tokens' not in instruction['args']:
            instruction['args']['max_tokens'] = 16
        if 'stop' not in instruction['args']:
            instruction['args']['stop'] = []
        if 'echo' not in instruction['args']:
            instruction['args']['echo'] = False
        if 'prompt_embedding' not in instruction['args']:
            instruction['args']['prompt_embedding'] = False
        if 'logprobs' not in instruction['args']:
            instruction['args']['logprobs'] = 0

        instruction['args']['seed'] = instruction.get('seed', 3406)
        job_id = instruction['id']
        if isinstance(job_id, str):
            job_id = [job_id]
        try:
            self.infer(job_id, instruction['args'])
        except Exception as e:
            traceback.print_exc()
            print("error in inference: "+str(e))

    def on_error(self, ws, msg):
        print(msg)

    def on_open(self, ws):
        ws.send(f"JOIN:{self.model_name}")

    async def on_message_other_worker(self, msg):
        print(f"<on_message_other_worker-{dist.get_rank()} recv: {msg}>")
        if msg.data.decode("utf-8") == 'START':
            self.infer(None, None)
        else:
            raise Exception("Unknown message")

    def start(self):
        my_rank = dist.get_rank()
        nats_url = os.environ.get("NATS_URL", "localhost:8092/my_coord")
        if my_rank == 0:
            async def listen():
                nc = await nats.connect(f"nats://{nats_url}")
                sub = await nc.subscribe(subject=self.model_name, queue=self.model_name, cb=self.on_message_prime_worker)
            loop = asyncio.get_event_loop()
            future = asyncio.Future()
            asyncio.ensure_future(listen())
            loop.run_forever()
        else:
            print("---")
            async def listen():
                nc = await nats.connect(f"nats://{nats_url}")
                sub = await nc.subscribe(subject=self.model_name+f".{self.model_id}", cb=self.on_message_other_worker)
            loop = asyncio.get_event_loop()
            future = asyncio.Future()
            asyncio.ensure_future(listen())
            loop.run_forever()
