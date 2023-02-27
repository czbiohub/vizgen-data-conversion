import click
from vizgen_data_module.experiment import VizgenExperiment, Timer

@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "-t",
    "--to",
    "output_format",
    default="starfish",
    help="Format to convert to (default=starfish).",
)
@click.option(
    "-e",
    "--experiment",
    "experiment_name",
    prompt="What is the experiment name?",
    help="Experiment name. There should be a directory with the same name inside both the raw and analysis folders.",
)
@click.option(
    "-r",
    "--raw-dir",
    "raw_path",
    prompt="Where is the raw data",
    help="Path to the directory housing the vizgen raw data. It should contain a subdirectory matching the experiment name.",
)
@click.option(
    "-a",
    "--analysis-dir",
    "analysis_path",
    prompt="Where is the analysis data",
    help="Path to the directory housing the vizgen analysis data. It should contain a subdirectory matching the expieriment name.",
)
@click.option(
    "-o",
    "--output",
    "out_path",
    prompt="Where should converted data be output to",
    help="Path to directory for outputting converted data.",
)
@click.option(
    "--tmp",
    "temp_path",
    prompt="Where can temporary conversion data be stored",
    help="Path to a directory for temporary data storage.",
)
def convert(
    output_format: str, experiment_name: str,
    raw_path: str, analysis_path: str,
    out_path: str, temp_path: str
):
    """Commandlfor converting vizgen data to starfish"""
    if output_format.lower() == "starfish":
        with Timer("Convert Vizgen to Starfish"):
            exp = VizgenExperiment(
                experiment_name=experiment_name,
                raw_path=raw_path,
                analysis_path=analysis_path
            )
            exp.export_to_starfish(out_path=out_path, temp_path=temp_path)
    else:
        print("Entered format is not supported. Please select from [starfish].")


@cli.command()
@click.option(
    "-e",
    "--experiment",
    "experiment_name",
    prompt="What is the experiment name?",
    help="Experiment name. There should be a directory with the same name inside both the raw and analysis folders.",
)
@click.option(
    "-r",
    "--raw-dir",
    "raw_path",
    prompt="Where is the raw data",
    help="Path to the directory housing the vizgen raw data. It should contain a subdirectory matching the experiment name.",
)
@click.option(
    "-a",
    "--analysis-dir",
    "analysis_path",
    prompt="Where is the analysis data",
    help="Path to the directory housing the vizgen analysis data. It should contain a subdirectory matching the expieriment name.",
)
@click.option(
    "-o",
    "--output",
    "out_path",
    prompt="Where should converted data be output to",
    help="Path to directory for outputting converted data.",
)
def export(
    experiment_name: str = None, raw_path: str = None,
    analysis_path: str = None, out_path: str = None
):
    """Command for exporting vizgen data to target dir
    """
    with Timer("Convert Vizgen to Starfish"):
        exp = VizgenExperiment(
            experiment_name=experiment_name,
            raw_path=raw_path,
            analysis_path=analysis_path
        )
        exp.export(out_path=out_path)

if __name__ == "__main__":
    cli()
