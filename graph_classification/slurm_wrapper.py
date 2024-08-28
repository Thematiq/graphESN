#!/usr/bin/python
import os
import sys

from argparse import ArgumentParser
from subprocess import Popen, PIPE, STDOUT


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', help='model name')
    parser.add_argument('-d', '--dataset', help='dataset name')
    parser.add_argument('-p', '--params', help='model params', required=False)
    parser.add_argument('-g', '--gpu', help='number of gpus to be used', type=int, required=False)
    parser.add_argument('-r', '--ram', help='amount of ram per cpu, with unit (e.g. 1GB)', type=str, required=False)
    parser.add_argument('-t', '--time', help='max time for SLURM', type=str, default='10:00:00')
    parser.add_argument('--mail-me', help='mail once the job has been done', action='store_true')
    parser.add_argument('--debug', help='run this script directly to debug', action='store_true')
    parser.add_argument('--inner-splits', type=int, default=10)
    parser.add_argument('--outer-splits', type=int, default=10)
    parser.add_argument('--partial-folds', nargs='*', default=None, type=str)

    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    params = args.params

    additional_resources = []
    if args.gpu is not None:
        additional_resources.append(f'gpu:{args.gpu}')

    if params is None:
        params = f"{model}.json"

    params_path = f"hyperparams/{params}"

    if args.partial_folds is not None:
        fold_arg = "/".join(args.partial_folds)
        filenames_fix = f"-{'_'.join(args.partial_folds)}"
    else:
        fold_arg = "None"
        filenames_fix = ""


    # output_name = f"logs/{model}-{dataset}{filenames_fix}.txt"
    output_name = "/dev/null"
    error_name = f"errors/{model}_v2-{dataset}{filenames_fix}.txt"
    csv_output = f"results/{model}_v2-{dataset}{filenames_fix}.csv"
    
    cmd = [
        f"--output={output_name}", 
        f"--error={error_name}", 
        f"--job-name={model}-{dataset}{filenames_fix}",
        f"--export=MODEL={model},DATASET={dataset},PARAMS={params_path},CSV={csv_output},IN_SPLITS={args.inner_splits},OUT_SPLITS={args.outer_splits},PARTIAL_FOLDS={fold_arg}",
        f"--time={args.time}",
    ] 

    if args.debug:
        cmd = [
            "./slurm_run_script.sh"
        ] + cmd

        env = os.environ.copy()
        # env = {}
        env["MODEL"] = f"{model}"
        env["DATASET"] = f"{dataset}"
        env["PARAMS"] = f"{params_path}"
        env["CSV"] = f"{csv_output}"
        env["IN_SPLITS"] = f"{args.inner_splits}"
        env["OUT_SPLITS"] = f"{args.outer_splits}"
        env["PARTIAL_FOLDS"] = f"{fold_arg}"
    else:
        cmd = ["sbatch"] + cmd
        env = None

    if len(additional_resources) > 0:
        cmd += [f'--gres={",".join(additional_resources)}']
    if args.mail_me:
        cmd += ["--mail-type=END,FAIL"]
    if args.ram is not None:
        cmd += [f"--mem-per-cpu={args.ram}"]

    if not args.debug:
        cmd += ["slurm_run_script.sh"]

    print(" ".join(cmd))

    # print(env)

    process = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
    stdout, stderr = process.communicate()

    print(stdout)
    print(stderr)


if __name__ == '__main__':
    main()
