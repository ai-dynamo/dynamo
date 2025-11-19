import os
import subprocess

CPUS_PER_GPU = 20
MEM_PER_CPU_GB = 11


def main():
    models = [
        ('meta-llama/Llama-3.1-8B-Instruct', 1),
        ('meta-llama/Llama-3.1-70B-Instruct', 4),
        ('meta-llama/Llama-3.1-70B-Instruct', 2),
        ('mistralai/Mixtral-8x7B-Instruct-v0.1', 2),
    ]
    num_passes = 1
    engines = ['tgi', 'vllm']
    for i in range(num_passes):
        for model in models:
            print(f"PASS {i} - Submitting job for {model[0]}")
            gpus = model[1]
            cpus_per_task = gpus * CPUS_PER_GPU
            for engine in engines:
                job_name = f'bench_{model[0].replace("/", "_")}_{engine}_pass_{i}'
                args = ['sbatch',
                        '--job-name', job_name,
                        '--output', f'/fsx/%u/logs/%x-%j.log',
                        '--time', '1:50:00',
                        '--qos', 'normal',
                        '--partition', 'hopper-prod',
                        '--gpus', str(gpus),
                        '--ntasks', '1',
                        '--cpus-per-task', str(cpus_per_task),
                        '--mem-per-cpu', str(MEM_PER_CPU_GB) + 'G',
                        '--nodes', '1',
                        ':',
                        '--gpus', '1',
                        '--ntasks', '1',
                        '--cpus-per-task', str(CPUS_PER_GPU),
                        '--mem-per-cpu', str(MEM_PER_CPU_GB) + 'G',
                        '--nodes', '1',
                        f'{engine}.slurm']
                env = os.environ.copy()
                env['MODEL'] = model[0]
                env['TP'] = str(gpus)
                process = subprocess.run(args, capture_output=True,
                                         env=env)
                print(process.stdout.decode())
                print(process.stderr.decode())
                if process.returncode != 0:
                    print(f'Error while submitting :: {args}')
                    exit(1)


if __name__ == '__main__':
    main()
