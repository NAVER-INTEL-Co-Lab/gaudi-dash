from typing import List
from functools import reduce
import random

random.seed(0)

train_types = ['warm', 'cold', 'dash', 'sp', 'reset', 'l2_init']
model = ['resnet18', 'vgg16']
optimizer_type = ['sgd', 'sam']
seed = 2021
session_per_once = 8

def iterate_list_of_list(lists: List[List[str]]):
    ptrs: List[int] = [0] * len(lists)
    ptrs_max = [len(cur_list) for cur_list in lists]
    ret_list = []
    tot_count = reduce(lambda x, y: x * y, ptrs_max)
    print("Total experiments:", tot_count)
    args_list = []
    for _ in range(tot_count):
        args_list.append([lists[i][cur] for i, cur in enumerate(ptrs)])
        last_check = len(lists) - 1
        carry = 1
        while last_check>=0:
            new_val = ptrs[last_check] + carry
            if ptrs[last_check] + carry >= ptrs_max[last_check]:
                ptrs[last_check] = new_val % ptrs_max[last_check]
                carry = new_val // ptrs_max[last_check]
                last_check -= 1
            else:
                ptrs[last_check] = new_val
                carry = 0
                break
        if carry == 1:
            print("Done")
    return args_list
    

iter_list = (iterate_list_of_list([train_types, model, optimizer_type]))

random.shuffle(iter_list)

print(len(iter_list))

def args_to_command(args):
    return f"PT_HPU_LAZY_MODE=1 python main.py --dataset cifar10 --model {args[1]} --train_type {args[0]} --optimizer_type {args[2]} --seed {seed}"

file_name = "run_at_once_2.sh"

with open(file_name, "w") as f:
    for i in range(0, session_per_once):
        commands = iter_list[16+i:16+i+1]
        commands = [args_to_command(command) for command in commands]
        commands = '; '.join(commands)
        with_screen = f'screen -dmS session_{i} bash -c "{commands}"'
        f.write(with_screen + "\n")

        
 

