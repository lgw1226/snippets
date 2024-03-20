'''
Sources
    wandb issue: https://github.com/wandb/wandb/issues/5641
'''

import wandb


if __name__ == '__main__':
    api = wandb.Api()
    entity = ''
    project = ''
    run_id = ''  # you can find wandb run id in wandb folder or online
    run = api.run(f"{entity}/{project}/{run_id}")
    for file in run.files():
        file.download()
        # you can also filter the file.download() based on the file name too
        # print(file)