


class FileLocParam:
    """
    A class representing a parameter that can be filled into a file path
    """
    def __init__(self, param_rep):
        self.param_rep = param_rep

    def __repr__(self):
        return self.param_rep


class ParameterizedFileLoc:
    """
    A filepath which can include parameters.
    Each allowable parameter should be defined below.
    """
    ENV = FileLocParam("<ENV>")
    DATE = FileLocParam("<DATE>")

    def __init__(self, path_element_list):
        """
        :param path_element_list: The list of elements defining the parameterized file loc.
                                    Each element in the list should be a string or one of the FileLocParam objects defined above.
        """
        self.path_element_list = path_element_list

    def fill(self, param, value):
        """
        Replace all occurrences of the parameter with a specific string value.

        :param param: The FileLocParam object to fill in
        :param value: The string value to fill in the parameter with.
        :return: A new ParameterizedFileLoc with the parameter filled in
        """
        return ParameterizedFileLoc([x if x != param else value for x in self.path_element_list])

    def __repr__(self):
        """
        :return: The string representation of the file loc.
        """
        return "".join(str(x) for x in self.path_element_list)


def parse_date(day_str, divider='-'):
    """
    Return a datetime object representing a string version of a date.

    :param day_str: The string version of the date
    :param divider: The divider between the year, month, and day.
    :return: The datetime object version of the date
    """
    if day_str is None:
        return None
    return datetime.datetime.strptime(day_str, divider.join(['%Y', '%m', '%d'])).date()


def format_day(day, divider):
    """
    Returns a string version of a datetime date.

    :param day: The datetime version of the date
    :param divider: The divider to use between the year, month, and day
    :return: The string version of the date.
    """
    return day.strftime(divider.join(['%Y', '%m', '%d']))


def get_bucket_name(gs_path):
    """
    Extracts the bucket name from a full GCS path

    :param gs_path: A full path in GCS, beginning with "gs://"
    :return: The first component of the path, which is the bucket name.
    """
    return re.match(r"^gs://([^/]+)/.*$", gs_path).group(1)


def split_bucket_name_and_blob(gs_path):
    """
    Splits a full GCS path into the bucket and blob

    :param gs_path: A full path in GCS, beginning with "gs://"
    :return: The bucket name, followed by the blob name.
    """
    bucket_name = get_bucket_name(gs_path)
    blob_path = gs_path[len("gs://") + len(bucket_name) + len("/"):]
    return bucket_name, blob_path


def transfer_local_file_to_storage(local_filepath, gs_path, project='hd-datascience-np'):
    """
    Takes a file stored on a VM or local machine and saves it at the given GCS location

    :param local_filepath: the path to the file locally
    :param gs_path: where the file should be stored in GCS
    :param project: The Google Cloud project that the file will be written to, defaults to "hd-datascience-np"
    """
    client = storage.Client(project=project)

    bucket_name, blob_path = split_bucket_name_and_blob(gs_path)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(local_filepath)


def transfer_storage_file_to_local(gs_path, local_filepath, project='hd-datascience-np'):
    """
    Copies a file stored in GCS to a VM or local machine

    :param gs_path: where the file is stored in GCS.
    :param local_filepath: where the file will be saved locally.
    :param project: The Google Cloud project that the file is stored in, defaults to "hd-datascience-np"
    """
    client = storage.Client(project=project)

    bucket_name, blob_path = split_bucket_name_and_blob(gs_path)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)

    blob.download_to_filename(local_filepath)


def get_most_recent_date(gs_path, max_date=None, project='hd-datascience-np', divider='-'):
    """
    Given a directory in GCS, this looks for the child directory whose name is a date and is the most recent.

    :param gs_path: The directory in GCS where we should be looking for date directories.
    :param max_date: Any date later than this parameter will be ignored.
    :param project: The Google Cloud project where the GCS path is located, defaults to "hd-datascience-np"
    :param divider: The divider between the year, month, and day in the date strings.
    :return: A datetime object representing the most recent date which has a directory, or None if none exist
    """
    client = storage.Client(project=project)

    bucket_name, prefix = split_bucket_name_and_blob(gs_path)
    bucket = client.get_bucket(bucket_name)

    dates = set()
    for b in bucket.list_blobs(prefix=prefix):
        m = re.match(r"^%s/(\d{4}-\d{2}-\d{2})(?:/.*)?$" % prefix, b.name)
        if m:
            dates.add(m.group(1))

    dates = list(dates)

    if max_date is not None:
        if type(max_date) is not str:
            max_date = max_date.strftime(divider.join(['%Y', '%m', '%d']))
        dates = [d for d in dates if d <= max_date]

    if len(dates) == 0:
        return None

    return datetime.datetime.strptime(max(dates), divider.join(['%Y', '%m', '%d'])).date()


def load_most_recent_df(df_load_fn, df_filepath, start_date):
    """
    This function takes in a filepath that is parameterized by a date, a function that loads a dataframe from the path,
    and a date to start at.  It finds the most recent date that has a corresponding dataframe.

    :param df_load_fn: a function that should take in a single argument, the filepath where the dataframe should be loaded from, and returns the read dataframe
    :param df_filepath: a ParameterizedFileLoc with all parameters filled except for for ParameterizedFileLoc.DATE which
                        will be filled in by the formatted date, e.g. gs://bucket/<DATE>/folder/to/load/*
    :param start_date: The first date that should be checked for a dataframe.
    :param date_format_fn: a function which should take in a single date argument and format it as a string.
                           This string will be used to fill in the filepath parameter.
    :param max_retries: the allowable number of dates which can fail before returning None.
                        Should be at least 0.
                        e.g. max_retries = 1 means this function should check the start date and the day before.
    :return: the read dataframe, and the day for which it was successfully read.
             If the max retries is reached, will return None for the dataframe and the start date for the date.
    """
    day = start_date

    df = df_load_fn(str(df_filepath.fill(ParameterizedFileLoc.DATE, day)))
    return df, day


