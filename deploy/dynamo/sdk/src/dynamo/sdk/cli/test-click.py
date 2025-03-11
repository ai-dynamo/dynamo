import click
import sys

@click.command()
@click.argument('command', nargs=-1)
def process(command):
    if command[0] == "dynamo" and command[1] == "run":
        args_str = ' '.join(command[2:])  # Join all args after "dynamo run"
        # Manually check for options
        options = []
        non_options = []
        for arg in command[2:]:
            if arg.startswith('-'):
                options.append(arg)
            else:
                non_options.append(arg)

        click.echo(f"Options: {options}")
        click.echo(f"Arguments: {' '.join(non_options)}")
    else:
        click.echo("Invalid command")

if __name__ == '__main__':
    process()