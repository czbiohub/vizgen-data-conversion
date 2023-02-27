import sys
import os
import time
from pathlib import Path, PosixPath
import json
import tempfile

from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import swifter
from slicedimage import ImageFormat

from vizgen_data_module.datareader import DaxReader
from vizgen_data_module.datawriter import TiffWriter

from starfish import Codebook
from starfish.types import Axes, Features
from starfish.experiment.builder import format_structured_dataset


def convert_dax_to_tiff_per_channel(experiment_name: str, raw_path: str, output_path: str) -> None:
    """Quickly create tiff files (per channel) from vizgen raw directory

    Args:
        raw_path (str): path to the vizgen raw folder
        output_dir (str): path to the target output folder
    """

    # Load the vizgen experiment
    experiment = VizgenExperiment(experiment_name, raw_path)
    # Group by FOV
    for fov, group in tqdm(experiment.data.groupby("fov"), desc="FOVs"):
        fov_dir = Path(output_path) / f"fov_{fov}"
        fov_dir.mkdir(parents=True, exist_ok=True)
        # Group by dax file so we only read each file once
        for dax_file, subgroup in group.groupby("dax"):
            dax_reader = DaxReader(dax_file)
            # Group by channel
            for channel, subsubgroup in subgroup.groupby("c"):
                # Define name for output tiff file
                tiff_path = (
                    "primary-images-fov_" + str(fov) + "-c" + str(channel) + ".tiff"
                )
                tiff_writer = TiffWriter(fov_dir / tiff_path)

                def add_frame(img, dax_reader, tiff_writer):
                    frame = dax_reader.loadAFrame(int(img.frame))
                    tiff_writer.addFrame(frame)

                subsubgroup.apply(
                    add_frame, dax_reader=dax_reader, tiff_writer=tiff_writer, axis=1
                )
                tiff_writer.close()


class InsufficientDiskSpaceError(Exception):
    "Raised when there is insufficient disk space for the operation"
    pass


class Timer(object):
    """ Wrapper object for timing functions.
    Usage:
        with Timer("process name"):
            func()
    """
    def __init__(self, name=None):
        self.name = name
        self.tstart = None

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(
                "[%s]" % self.name,
            )
        print(f"Elapsed: {(time.time() - self.tstart)/60:.2f} minutes.")


class VizgenExperiment:
    """Object for loading and handling Vizgen Experiments

    Stores:
        self.codebook:  experiment codebook
        self.data:      experiment data (metadata) dataframe
    Primary Functions:
        export_to_starfish: converts to a starfish-compatible format
    """

    def __init__(self, experiment_name: str, raw_path: str, analysis_path: str = None) -> None:
        """Initialize

        Args:
            raw_path (str): Path to raw data directory
            analysis_path (str): Path to raw analysis directory
        """
        print("\nInitializing vizgen experiment...")
        self.name = experiment_name
        # Raw Folder paths
        self.raw_path: PosixPath = Path(raw_path)/self.name
        self.codebook_csv: PosixPath = list(self.raw_path.glob("codebook*.csv"))[0]
        self.data_csv: PosixPath = self.raw_path / "dataorganization.csv"
        self.data_path: PosixPath = self.raw_path / "data"

        # Analysis folder paths
        self.u2p_matrix: np.array = None
        if analysis_path:
            self.viz_analysis: PosixPath = Path(analysis_path)/self.name
            self.matrix_file: PosixPath = list(
                self.viz_analysis.glob("**/micron_to_mosaic_pixel_transform.csv")
            )[0]
            # Micron to Pixel matrix
            self.u2p_matrix = np.loadtxt(self.matrix_file)

        self.n_channels = 3  # Assuming 3 channels

        # Initialize object variables
        self.rc_map = {}
        self.inf_files = []
        self.dax_files = []

        self._codebook = None
        self._vizgen_data = None
        self._data = None

    @property
    def codebook(self):
        """Cobebook accessor. If uninitialized, load from file, then return."""
        if self._codebook is None:
            self._codebook = self._load_csv_codebook(self.codebook_csv)
        return self._codebook

    @property
    def vizgen_data(self):
        """Vizgen csv accessor. If uninitialized, load from file, then return."""
        if self._vizgen_data is None:
            self._vizgen_data = self._load_vizgen_data(self.data_csv)
        return self._vizgen_data

    @property
    def data(self):
        """Exp. data accessor. If uninitialized, load from file, then return."""
        if self._data is None:
            self._data = self._load_experiment_data(self.data_path, self.vizgen_data)
        return self._data

    def read_inf(self, inf_file: str) -> dict:
        """Read an inf file and return the metadata

        Args:
            inf_file (str): path to .inf file

        Returns:
            dict: Dictionary of file metadata
        """
        with open(inf_file, encoding="UTF-8") as file:
            lines = file.readlines()
            return dict(line.rstrip().split(" = ") for line in lines)

    def _load_csv_codebook(self, codebook_path: PosixPath = None) -> Codebook:
        """Load the vizgen codebook using

        Args:
            csv_path (PosixPath): Path to vizgen codebook.csv

        Returns:
            Starfish.Codebook: Returns a starfish-formatted codebook dataframe
        """
        print("Loading vizgen codebook...")
        # Enforce default path
        codebook_path = self.codebook_csv if not codebook_path else codebook_path

        # Load csv into dataframe
        csv_codebook: pd.DataFrame = pd.read_csv(codebook_path, index_col=0)
        csv_codebook = csv_codebook.drop(columns=["id"])  # Drop ID column

        # Each bitcode name corresponds to a (round, channel) combination
        # Define a mapping to unpack (r, c) from table row
        self.rc_map = {
            csv_codebook.columns[i]: (i // self.n_channels, i % self.n_channels)
            for i in range(csv_codebook.shape[1])
        }

        def parse_codewords(row: pd.Series) -> dict:
            """Use self.rc_map to unpack r, c, codeword from vizgen csv row
                Designed to be used with csv_codebook.apply(sef.parse_codewords)

            Args:
                row (pd.Series): Row entry in vizgen csv table

            Returns:
                dict: Starfish mappings
            """
            row = row[row == 1]
            codewords = [
                {Axes.ROUND.value: r, Axes.CH.value: c, Features.CODE_VALUE: 1}
                for (r, c) in (self.rc_map[marker] for marker in row.index)
            ]
            return {Features.CODEWORD: codewords, Features.TARGET: row.name}

        # Unpack csv_codebook using mapping and convert to starfish dataframe
        mappings = csv_codebook.apply(parse_codewords, axis=1).to_list()
        return Codebook.from_code_array(mappings)

    def _load_vizgen_data(self, csv_path: PosixPath) -> pd.DataFrame:
        """Load vizgen dataorganization.csv and process

        Args:
            csv_path (PosixPath): Path to dataorganization.csv

        Returns:
            pd.DataFrame: Dataframe containing processed csv data
        """
        print("Loading vizgen data organization csv...")
        # Enforce default path
        csv_path = self.data_csv if not csv_path else csv_path

        # Read CSV file to dataframe
        df = pd.read_csv(csv_path)
        # Extract z-position and frame column values
        df.zPos, df.frame = (
            df.zPos.apply(lambda x: x[1:-1].split(",")),
            df.frame.apply(lambda x: x[1:-1].split(",")),
        )
        return df

    def _load_experiment_data(
        self, data_path: PosixPath, vizgen_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Load experimental data to a dataframe using dataorganization.csv

        Args:
            data_path (PosixPath): path to the raw data directory
            dataorganization_csv (PosixPath): path to dataorganization.csv

        Returns:
            pd.DataFrame: dataframe housing all necessary dataset metadata
        """
        print("Loading vizgen experiment data...")
        # Enforce default paths
        data_path = self.data_path if data_path is None else data_path
        vizgen_data = self.vizgen_data if vizgen_data is None else vizgen_data

        # Index raw directory for dax and inf files
        self.inf_files = list(sorted(data_path.glob("*.inf")))
        self.dax_files = list(sorted(data_path.glob("*.dax")))

        # Construct a df of images with their corresponding inf/dax paths
        file_names = [f.stem for f in self.dax_files]
        df = pd.DataFrame(
            {"image": file_names, "dax": self.dax_files, "inf": self.inf_files}
        )
        # Extract stack, round, fov from file name
        df[["modality", "round", "fov"]] = df.image.str.split("_", expand=True)
        # Read .inf metadata for each image and concat into dataframe
        df = pd.concat([df, pd.DataFrame(df.inf.apply(self.read_inf).tolist())], axis=1)

        # Split images/rows into unique frames/z-positions
        data_organization = vizgen_data.apply(
            lambda x: x.explode() if x.name in ["frame", "zPos"] else x
        ).reset_index(drop=True)

        # reformat "round" for merging
        df.loc[df["round"] == "prestain", ["round"]] = -1
        df["round"] = df["round"].astype(int)

        # Merge filepath + .inf metadata with dataorganization.csv
        df = pd.merge(
            df, data_organization, left_on="round", right_on="imagingRound", how="left"
        )

        # Define a modality channel for image type
        df["modality"] = df["channelName"]
        df.loc[df["round"] != -1, ["modality"]] = "primary"

        # revert "round" column after merge
        df.loc[df["round"] == -1, ["round"]] = df["round"].astype(int).max() + 1

        def index_map(values: pd.Series) -> dict:
            """Create a dict/map that converts a list of values to a list of
                int indices
                Can be created for an entire column, then applied to each row
                ex. [560, 650, 750] -> [0, 1, 2]

            Args:
                values (pd.Series): A pd.Series representing a list of values

            Returns:
                dict: a dictionary mapping unique values to unique indices {v,i}
            """
            unique_values = values.unique().astype(float)
            indices = list(range(len(unique_values)))
            return dict(zip(unique_values, indices))

        # Create maps to unique indices for colors and z-positions
        color_map = index_map(df.color)
        z_pos_map = index_map(df.zPos)

        if self.u2p_matrix is not None:
            # Define image tile position in microns (convert from pixel positions)
            start_pixels = (
                df[["x_start", "y_start", "zPos"]]  # Get top-left corner
                .astype(float)
                .to_numpy()
            )
            # pd.DataFrame(df2.teams.tolist(), index= df2.index)
            df[["x_min", "y_min", "z_min"]] = pd.DataFrame(
                [  # convert to microns using u2p_matrix
                    row for row in (np.linalg.inv(self.u2p_matrix) @ start_pixels.T).T
                ]
            )
            end_pixels = (
                df[["x_end", "y_end", "zPos"]]  # Get bottom-right corner in pixels
                .astype(float)
                .to_numpy()
            )
            df[["x_max", "y_max", "z_max"]] = pd.DataFrame(
                [  # convert to microns using u2p_matrix
                    row for row in (np.linalg.inv(self.u2p_matrix) @ end_pixels.T).T
                ]
            )

        # Define (r, c, z) columns using index maps where needed
        df["r"] = df["round"]
        df["c"] = df.color.apply(lambda x: color_map[x])
        df["z"] = df.zPos.apply(lambda x: z_pos_map[float(x)])

        # Define name for output tiff file
        df["tiff"] = (
            df.modality.astype(str)
            + "-f"
            + df.fov.astype(str)
            + "-r"
            + df.r.astype(str)
            + "-c"
            + df.c.astype(str)
            + "-z"
            + df.z.astype(str)
            + ".tiff"
        )

        return df

    def _create_csv(self, data: pd.DataFrame, temp_path: str):
        """create coordinates.csv in a target dir

        Args:
            data: subgroup from self.data to export
            temp_path: path to target temp directory for output
        """
        csv_data = data[
            ["fov", "r", "c", "z", "x_min", "y_min", "z_min", "x_max", "y_max", "z_max"]
        ]
        csv_data.columns = [
            "fov",
            "round",
            "ch",
            "zplane",
            "xc_min",
            "yc_min",
            "zc_min",
            "xc_max",
            "yc_max",
            "zc_max",
        ]
        csv_data.to_csv(Path(temp_path) / "coordinates.csv", header=True)

    def _export_dax_frame_to_tiff(
        self, dax_reader: DaxReader, frame_n: int, out_path: str, file_name: str
    ):
        """Write a target frame from a DaxReader object to file

        Args:
            dax_reader: object for reading an existing dax file
            frame_n: the target frame to write
            out_path: path to the target output directory
        """
        out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / file_name

        data = dax_reader.loadAFrame(frame_n)
        tiff_file = TiffWriter(out_path)
        tiff_file.addFrame(data)
        tiff_file.close()

    def _export_dax_to_tiff(self, dax_frames, dax_path, out_path):
        dax_reader = DaxReader(dax_path)
        dax_frames.loc[dax_frames.dax == dax_path, "sha256"] = dax_reader.hashID()
        dax_frames.swifter.progress_bar(False).apply(
            lambda x: self._export_dax_frame_to_tiff(
                dax_reader=dax_reader,
                frame_n=int(x.frame),
                out_path=out_path,
                file_name=x.tiff,
            ),
            axis=1,
        )

    def _export_image_data(self, data: pd.DataFrame, temp_path: str):
        """Export dax image data to tiff frames in the target directory

        Args:
            data: subgroup from self.data to convert and export
            temp_path: targout output tempoary directory path
        """
        # export image slices by group so we only have to read each dax once
        data.swifter.groupby("dax").progress_bar(
            enable=True, desc=f"Converting dax files to tiff"
        ).apply(lambda x: self._export_dax_to_tiff(x, x.name, temp_path))

    def _export_data(self, data: pd.DataFrame, temp_path: str):
        """export group data to target directory

        Args:
            data: subgroup from self.data to convert and export
            temp_path: target output temporary directory
        """
        # export coordinates data to a temp csv
        self._create_csv(data=data, temp_path=temp_path)
        # convert and export dax images to tiff in a temporary dir
        self._export_image_data(data=data, temp_path=temp_path)

    def _get_dir_size(self, dir_path: str):
        root = Path(dir_path)
        return sum(f.stat().st_size for f in root.glob("**/*") if f.is_file())

    def _get_available_space(self, dir_path: str):
        with tempfile.TemporaryDirectory(dir=dir_path) as tmp:
            statvfs = os.statvfs(tmp)
            return statvfs.f_frsize * statvfs.f_bfree

    def _check_available_space(self, target_path: str):
        """Raise an error if there isn't sufficient space to write the new data
        """
        dir_size = self._get_dir_size(self.raw_path)
        available_space = self._get_available_space(target_path)

        if dir_size >= available_space:
            msg = (
                f"\nERROR: Insufficient disk space available for additional "
                f"data at {target_path}. Please provide an alternate path"
                f"on a disk with more free space than the raw data size.\n"
            )
            raise InsufficientDiskSpaceError(msg)

    def _merge_experiment_files(self, out_path):
        out_path = Path(out_path)
        with open(out_path / "primary" / "experiment.json", "r") as f:
            experiment = json.load(f)
        for subdir in out_path.glob("*"):
            if subdir.is_dir():
                img_type = str(subdir.name)
                experiment["images"][img_type] = f"{img_type}/{img_type}.json"
                # (subdir/"experiment.json").unlink()
                # (subdir/"codebook.json").unlink()
        # save experiment
        with open(out_path / "experiment.json", "w") as f:
            json.dump(experiment, f, indent=4)
        # save codebook
        self.codebook.to_json(out_path / "codebook.json")


    def export(self, out_path: str) -> None:
        """export all data to target directory
        """
        try:
            # Make parent directory and validate there's enough disk space
            Path(out_path).mkdir(parents=True)
            self._check_available_space(out_path)

            # Convert each data type separately (primary, auxilliary)
            for name, group in self.data.groupby("modality"):
                # Make output directory
                sub_dir = (Path(out_path) / name)
                sub_dir.mkdir(parents=True)
                print(f"\nMigrating {name} data...")
                # Export the data data
                self._export_data(group, sub_dir)

            print("\nConversion complete!\n")

        except InsufficientDiskSpaceError as err:
            print(err)
        except FileExistsError:
            print(
                "Output directory already exists. Please delete or choose a"
                "different directory."
            )


    def export_to_starfish(self, out_path: str, temp_path: str = None):
        """export experiment data to a starfish directory

        Args:
            out_path: export final starfish data to out_path
            temp_path: path where to create a temporary directory for
                        intermediary data creation. be sure to select path with
                        read/write permission and sufficient disk space
        """
        try:
            # Make parent directory and validate there's enough disk space
            Path(out_path).mkdir(parents=True)
            self._check_available_space(temp_path)
            self._check_available_space(out_path)

            # Convert each data type separately (primary, auxilliary)
            for name, group in self.data.groupby("modality"):
                print(f"\nMigrating {name} data...")
                # Create temporary dir to house intermediary data conversion
                with tempfile.TemporaryDirectory(dir=temp_path) as temp_dir:
                    # Export the intermediary data
                    self._export_data(group, temp_dir)
                    # Make output directory
                    (Path(out_path) / name).mkdir(parents=True)
                    # Convert intermediary data to starfish format
                    print(f"Converting {name} files to starfish...")
                    with Timer("Starfish format_structured_data"):
                        format_structured_dataset(
                            temp_dir,
                            Path(temp_dir) / "coordinates.csv",
                            Path(out_path) / name,
                            ImageFormat.TIFF,
                        )

            # merge experiment.json files for each data type
            self._merge_experiment_files(out_path)

            print("\nConversion complete!\n")

        except InsufficientDiskSpaceError as err:
            print(err)
        except FileExistsError:
            print(
                "Output directory already exists. Please delete or choose a"
                "different directory."
            )


if __name__ == "__main__":
    pass
