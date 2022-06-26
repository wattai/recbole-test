
import click
from recbole.quick_start import run_recbole

@click.command()
@click.option(
    "-d",
    "--dataset_name",
    required=True,
    type=str,
    help="Dataset Name(your custom dataset name or recbole's dataset name)",
)
@click.option(
    "-c",
    "--config_file",
    required=True,
    type=str,
    help="config file path",
)
def main(dataset_name, config_file):
    config_file_list = [config_file]

    run_recbole(
        model="xDeepFM",
        dataset=dataset_name,
        config_file_list=config_file_list,
    )

if __name__ == "__main__":
    main()

