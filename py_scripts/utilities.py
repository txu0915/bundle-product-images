from google.cloud import storage
import re
import datetime
from datetime import timedelta
from py4j.protocol import Py4JError
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import os
import argparse
import json
import os
import json
import pyspark.sql.functions as F
import tempfile
import skimage.io as skio
import os
import time
from skimage.color import rgb2gray
from skimage import filters
from scipy import ndimage as ndi
from skimage import morphology
import numpy as np
from skimage.transform import resize,rescale
from PIL import Image
from py_scripts.model_mart_file_utils import *
from py_scripts.model_mart_img_utils import *
print("imported")
## general utils...
def get_layout(config, bundle_type):
    """
    Gets the specified bundle type from the config and returns the object containing the bundle's layers info
    :param config: The parsed config object with all bundles
    :param bundle_type: The specific bundle type we are looking for
    :return: The bundle object definition.
    """
    for bundle in config:
        if bundle['bundle_type'] == bundle_type:
            return bundle
    return None


def filter_product_list_df(product_list_df):
    """
    There may be multiple products with the same product type in the product list.
    Filter each list to contain each product type only once.

    :param product_list_df: The input df with columns (line_key, product_list_json)
    :return: df with the same two columns as the input (line_key, product_list_json)
    """

    def filter_product_list_to_single_sku_per_product_type(product_list_json):
        """
        Parse the input product list String, and filter it so that each product type only appears once.

        :param product_list_json: a JSON string representation of the product list
        :return: a JSON string representation of the filtered product list
        """
        prod_list = json.loads(product_list_json)
        filtered_list = []
        product_types = set()

        for prod in prod_list:
            if prod['productType'] not in product_types:
                filtered_list.append(prod)
                product_types.add(prod['productType'])
        # return ".".join(filtered_list)
        return json.dumps(filtered_list)


    filter_list_udf = F.udf(filter_product_list_to_single_sku_per_product_type, StringType())
    return product_list_df.select('line_key', filter_list_udf('product_list_json').alias('product_list_json'))



def get_layer_skus_df(product_lists_df):
    """
    Take all the product lists defining the bundles, and get all unique oms_ids from those lists
    with the associated product type of each.

    :param product_lists_df:
    :return: df with two columns: (oms_id, product_type)
    """
    return product_lists_df.select(
        F.explode(F.from_json('product_list_json', ArrayType(
            StructType([
                StructField("id", StringType()),
                StructField("productType", StringType())
            ])
        ))
                  ).alias('product_object')) \
        .select(F.col('product_object.id').alias('oms_id'), F.col('product_object.productType').alias('productType')) \
        .drop_duplicates()


def get_idm_attributes(spark, day, idm_attributes_file):
    """
    Load the latest IDM attributes data starting with the specified day and working backwards until the data is found.
    :param spark: SparkSession to read data
    :param day: The data to start looking for IDM data from.
    :return: df with columns (oms_id, attribute_id, attribute_value)
    """
    def load_df(idm_attr_loc):
        schema = [
            StructField('oms_id', StringType()),
            StructField('attribute_id', StringType()),
            StructField('attribute_value', StringType()),
            StructField('x1', StringType()),
            StructField('x2', StringType()),
            StructField('x3', StringType()),
            StructField('x4', StringType())
        ]
        return spark.read.csv(idm_attr_loc, schema=StructType(schema), sep=',')

    #idm_attributes_df, _ = load_most_recent_df(load_df, idm_attributes_file, day, lambda d: format_day(d, '-'))
    #idm_attributes_df = load_df(str(IDM_ATTRIBUTES_LOC.fill(ParameterizedFileLoc.DATE, "10-20-2020")))
    idm_attributes_df = load_df(idm_attributes_file)

    return idm_attributes_df.select('oms_id', 'attribute_id', 'attribute_value')

IDM_ATTRIBUTES_LOC = ParameterizedFileLoc(["gs://hd-personalization-prod-data/MergedFullFeed/",
                                           ParameterizedFileLoc.DATE,
                                           "/ItemAttributes/*"])



def get_sku_img_df(layer_skus_df, day, spark, idm_attributes_file):
    """
    For each oms_id that is in a bundle, we need to know which image to use to represent the SKU.
    Load the IDM attributes data, filter to keep only the primary image for each SKU,
    and join with the skus appearing in a bundle.

    :param layer_skus_df: The df of skus appearing in a bundle.
    :param day: The day that this job is running, used to define which day's IDM data to load
    :param spark: SparkSession to read data
    :return: df with columns: (oms_id, productType, img_guid)
    """
    return get_idm_attributes(spark, day, idm_attributes_file) \
        .where(F.col('attribute_id') == "645296e8-a910-43c3-803c-b51d3f1d4a89") \
        .select('oms_id', F.col('attribute_value').alias('img_guid')) \
        .join(layer_skus_df, how='inner', on='oms_id')    #

def create_layer_img(img, image_config, layer):
    """
    Given the product image, segment out the foreground from the background,
    then scale and move the foreground based on the specifications from the layer info.

    :param img: The product image, stored as a numpy array (3 channels)
    :param image_config: config giving specifics about drawing the images.
    :param layer: The layer specifications for the specific product type
    :return: The layer image, stored as a numpy array (4 channels including alpha)
    """
    composite_img_size = max(image_config['image_sizes'])
    erode_size = image_config['erode_size']
    bottom_border = image_config['bottom_border']

    ulx = int(composite_img_size * layer['ulx'])
    uly = int(composite_img_size * layer['uly'])
    w = int(composite_img_size * layer['w'])

    # create the segmentation mask for the product
    silhouette = get_silhouette_mask(img, image_config)

    # get the bounding box around the foreground, eroded by the erode size
    silhouette_y, silhouette_x = silhouette.nonzero()
    min_x = min(silhouette_x) + erode_size
    max_x = max(silhouette_x) - erode_size
    min_y = min(silhouette_y) + erode_size
    max_y = max(silhouette_y) - erode_size

    delta_x = max_x - min_x
    delta_y = max_y - min_y

    # make sure it won't extend off bottom of screen
    h_scaled = 1. * delta_y * w / delta_x
    if h_scaled + uly > composite_img_size - bottom_border:
        scale_factor = 1. * (composite_img_size - bottom_border - uly) / delta_y
    else:
        scale_factor = 1. * w / delta_x

    # scale the foreground to the appropriate size
    prod_img = rescale(img[min_y: max_y + 1, min_x: max_x + 1], scale_factor, mode='reflect', multichannel=True,
                       anti_aliasing=True, preserve_range=True)

    # scale the segmentation mask to the appropriate size
    silhouette = 255 * silhouette
    silhouette = rescale(silhouette[min_y: max_y + 1, min_x: max_x + 1], scale_factor, mode='reflect',
                         multichannel=False, anti_aliasing=True, preserve_range=True)

    prod_img = Image.fromarray(prod_img.astype(np.uint8))
    silhouette = Image.fromarray(silhouette.astype(np.uint8))

    # create the transparent product image layer, and paste the scaled product image into it,
    # using the mask to only include the foreground.
    layer_img = Image.new(size=(composite_img_size, composite_img_size), mode="RGBA", color=(255, 255, 255, 0))
    layer_img.paste(prod_img, box=(ulx, uly), mask=silhouette)

    return np.array(layer_img)


def create_and_save_layer_imgs(sku_img_df, kitchen_package_layout, image_config, img_dir,
                               local_layer_folder_name="layers"):
    """
    For each sku, we know what type of product it is and what image to use.
    Download the image, create a transparent background image layer for that SKU, and save the layer locally and in GCS.

    :param sku_img_df: df with columns (oms_id, product_type, img_guid)
    :param kitchen_package_layout: config giving the size and location for each product type
    :param image_config: config giving specifics about drawing the images.
    :param img_dir: Where the layer images will be stored in GCS
    :param local_layer_folder_name: Where the layer images will be stored locally.
    :return: the folder where the layer images are stored locally.
    """
    if not os.path.isdir(local_layer_folder_name):
        os.makedirs(local_layer_folder_name)

    #sku_img_pd_df = sku_img_df.toPandas()

    for row in sku_img_df.rdd.collect():

    #for i in sku_img_pd_df.index:
        #row = sku_img_pd_df.loc[i]

        img = get_image(row.img_guid)

        layer_img = create_layer_img(img, image_config, [x for x in kitchen_package_layout['bundle_layers'] if
                                                         x['productType'] == row.productType][0])

        save_img(layer_img, "%s.png" % row.oms_id,
                 "%s/%s/%s" % (img_dir, format_day(datetime.datetime.today(), '-'), LAYER_IMGS_SUFFIX),
                 local_dir=local_layer_folder_name,
                 image_sizes=image_config.get("image_sizes", None))

    return local_layer_folder_name


def get_all_product_sets_df(product_lists_df, kitchen_package_layout):
    """
    Each bundle can potentially contain multiple products with the same product type.
    We want to get all the possible sets of products where there is one product for each product type.
    These sets will be used to combine the product image layers into the full bundle images.

    :param product_lists_df: input df with all the products for a specific bundle (line_key, product_list_json)
    :param kitchen_package_layout: The layer definitions for the specified bundle type
    :return: df with columns (line_key, img_id, sku_list)
    """

    def create_all_product_sets(prod_list_json, kitchen_package_layout):
        """
        Given the full list of products and their product types,
        as well as the definition of which product types belong in a bundle,
        create a list with all the subsets of the bundle products, where there is only a single product of each product type

        :param prod_list_json: The json string representing all products that belong to this bundle.
        :param kitchen_package_layout: The definition of the layers in the bundle.
        :return: a list of dicts, each containing a sku list "sku_list" and an id "img_key".  The sku list is the order
                the product layers should be stacked.
        """
        product_types = [layer['productType'] for layer in kitchen_package_layout['bundle_layers']]
        prod_type_to_skus = {pt: [] for pt in product_types}
        # print(product_types)
        # print(prod_type_to_skus)
        for prod in json.loads(prod_list_json):
            prod_type_to_skus[prod['productType']].append(prod['id'])
        # print(prod_type_to_skus)
        # print("product_types:\n")
        # print(product_types)
        all_sets = [[]]
        for pt in product_types:
            next_all_sets = []
            if prod_type_to_skus[pt] != []:
                for oms_id in prod_type_to_skus[pt]:
                    for curr_set in all_sets:
                        next_all_sets.append(curr_set + [oms_id])
                    print(oms_id, "next_all_sets:", next_all_sets)
                all_sets = next_all_sets
            print("all_sets_in_loop:", all_sets)
        print("allsets:", all_sets)
        return [{"sku_list": all_sets[i], 'img_key': str(i)} for i in range(len(all_sets))]

    all_product_sets_udf = F.udf(lambda p: create_all_product_sets(p, kitchen_package_layout),
                                 ArrayType(
                                     StructType([
                                         StructField('sku_list', ArrayType(StringType())),
                                         StructField('img_key', StringType())
                                     ])
                                 ))

    return product_lists_df.select('line_key', F.explode(all_product_sets_udf('product_list_json')).alias('skus_key')) \
        .select('line_key', F.concat_ws('_', F.col('line_key'), F.col('skus_key.img_key')).alias('img_id'), 'skus_key.sku_list')



def create_package_img(sku_list, image_config, layer_img_folder):
    """
    Combine the layers for the specified SKUs into a single bundle image.

    :param sku_list: The list of oms_ids that need layers combined into the bundle image
    :param image_config: config giving specifics about drawing the images.
    :param layer_img_folder: The directory on the local machine where all the individual product layers are saved
    :return: The composited bundle image, as a numpy array with 3 channels
    """
    composite_img_size = max(image_config['image_sizes'])
    package_img = Image.new(size=(composite_img_size, composite_img_size), mode="RGBA", color=(255, 255, 255, 255))

    for oms_id in sku_list:
        layer = Image.open(os.path.join(layer_img_folder, "%s_%d.png" % (oms_id, composite_img_size)))
        package_img = Image.alpha_composite(package_img, layer)

    return np.array(package_img.convert(mode="RGB"))


def create_and_save_package_imgs(all_product_sets_df, layer_img_folder, image_config, img_dir):
    """
    Given the set of products which are going to be combined into a bundle image, create the bundle by stacking the
    individual product image layers, then save the image to GCS.

    :param all_product_sets_df: the input data with columns (line_key, img_id, sku_list)
    :param layer_img_folder: The directory on the local machine where all the individual product layers are saved
    :param image_config: config giving specifics about drawing the images.
    :param img_dir: The location in GCS where the bundle images will be saved
    """
    #all_product_sets_pd_df = all_product_sets_df.toPandas()

    bundle_image_filetype = image_config["bundle_image_filetype"]
    for row in all_product_sets_df.rdd.collect():
    #for i in all_product_sets_pd_df.index:
        #row = all_product_sets_pd_df.loc[i]

        package_img = create_package_img(row.sku_list, image_config, layer_img_folder)

        save_img(package_img, "%s.%s" % (row.img_id, bundle_image_filetype),
                 "%s/%s/%s" % (img_dir, format_day(datetime.datetime.today(), '-'), PACKAGE_IMGS_SUFFIX),
                 image_sizes=image_config.get("image_sizes", None))


def save_package_image_ids_mapping(ids_df, package_dir, day):
    """
    Save a file to GCS giving the mapping from the original package id to the image id.

    :param ids_df: df with columns (line_key, img_id) that will be saved
    :param package_dir: The location where the package details are stored in GCS
    :param day: The day that the data being processed is from.
    """
    ids_df.toPandas()[['line_key', 'img_id']].to_csv(PACKAGE_TO_IMAGE_ID_MAP_FILENAME, index=False, header=False)
    transfer_local_file_to_storage(PACKAGE_TO_IMAGE_ID_MAP_FILENAME,
                                   "%s/%s/%s" % (package_dir, format_day(day, '-'), PACKAGE_TO_IMAGE_ID_MAP_FILENAME))


def create_dir_if_needed(filepath):
    dir, _ = os.path.split(filepath)
    if not os.path.isdir(dir):
        os.makedirs(dir)


def get_config(config_filepath, temp_local_filepath='/tmp/config.json'):
    """
    Loads the json config file from the specified location in GCS
    :param config_filepath: The storage location in GCS
    :param temp_local_filepath: where the file is copied to locally before being parsed.
    :return: The parsed config object
    """
    #create_dir_if_needed(temp_local_filepath)

    #transfer_storage_file_to_local(config_filepath, temp_local_filepath)

    with open(config_filepath, 'r') as fp:
        return json.load(fp)


def get_product_list_df(input_filepath, day, spark):
    """
    Reads the package definition file and begins parsing it
    :param package_dir: Where the package definition file is located
    :param day: The specific day that we are going to load
    :param spark: SparkSession to read the data
    :return: df with two columns (line_key, product_list_json)
    """
    local_filepath = "./Data/temp.txt"
    create_dir_if_needed(local_filepath)
    transfer_storage_file_to_local(input_filepath, local_filepath, project='hd-datascience-np')
    return spark.read.text(local_filepath) \
                .select(F.split(F.col('value'), "~!~").alias("split_value")) \
                .select(F.col("split_value").getItem(0).alias("line_key"), F.col("split_value").getItem(1).alias("product_list_json"))


## model mart file utils...
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

def load_most_recent_df(df_load_fn, df_filepath, start_date, date_format_fn, max_retries=30):
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
    retries = 0
    day = start_date

    while retries <= max_retries:
        try:
            df = df_load_fn(str(df_filepath))
            df.take(1)  # to force it to load the data now and fail early if the file doesn't exist

            return df, day
        except Py4JError as e:
            pass
        retries += 1
        day -= timedelta(days=1)

    return None, start_date

## Model mart image utils...



def create_img_cache():
    """
    :return: a temporary directory which images can be saved in.
    """
    return tempfile.TemporaryDirectory()


def clear_img_cache(cache):
    """
    :param cache: deletes the temporary directory and any images stored in it.
    """
    cache.cleanup()


def get_image(img_input, cache=None, download_sleep_time=1, num_retries=2):
    """
    Retrieves the image from the specified guid.

    :param img_input: The guid for the image.  If it doesn't have a file extension, will assumed to be .jpg
    :param cache: A cached to look for images in.  If the image has already been downloaded and saved in the cache,
                    this will be much faster than redownloading it.
    :param download_sleep_time: How long to wait after downloading a new image to prevent spamming servers with calls.
                                Defaults to 1.0 sec
    :param num_retries: How many times to retry downloading an image if it fails. Should be 0 or larger, defaults to 2.
    :return: the image as a numpy array.
    """
    _, img_guid_maybe_ext = os.path.split(img_input)
    img_guid, ext = os.path.splitext(img_guid_maybe_ext)
    if ext is None or len(ext) == 0:
        ext = ".jpg"

    img_filename = img_guid + ext

    if cache is not None and os.path.exists(os.path.join(cache.name, img_filename)):
        return skio.imread(os.path.join(cache.name, img_filename))

    url = "http://idm.homedepot.com/assets/image/%s/%s" % (img_filename[:2], img_filename)

    while num_retries >= 0:
        try:
            img = skio.imread(url)
            break
        except Exception as e:
            print("Couldn't read image from url %s" % url)
            time.sleep(download_sleep_time)
            num_retries -= 1
            img = None

    if img is None:
        return None

    if download_sleep_time is not None and download_sleep_time > 0:
        time.sleep(download_sleep_time)

    if cache is not None:
        skio.imsave(os.path.join(cache.name, img_filename), img)

    return img


def save_img(img, filename, storage_dir, local_filename=None, local_dir=None, image_sizes=None):
    """
    Saves an image to GCS.

    :param img: the image to be saved, as a numpy array
    :param filename: the filename that the image will be saved as in storage.
    :param storage_dir: the directory in GCS where the image will be saved.
    :param local_filename: If provided, the image will be saved with the given filename before transferring to GCS.
                            Defaults to None, which means the image will be saved with the same filename as in GCS.
    :param local_dir: a local directory where the image should be saved before transferring to GCS, defaults to None
    :param image_sizes: a list of sizes for the image. Defaults to None.  If provided, the image will be resized to
                        be NxN square and saved as filename_size.ext for each N in the list.
    """
    def save(img, filename, storage_dir, local_filename, local_dir):
        if local_filename is None:
            local_filename = filename

        if local_dir is not None:
            local_filepath = os.path.join(local_dir, local_filename)
        else:
            local_filepath = local_filename

        skio.imsave(local_filepath, img)
        transfer_local_file_to_storage(local_filepath, os.path.join(storage_dir, filename))

    if image_sizes is None:
        save(img, filename, storage_dir, local_filename, local_dir)
    else:
        filename_no_ext, ext = os.path.splitext(filename)

        for sz in image_sizes:
            i = resize(img, (sz, sz), mode='reflect', anti_aliasing=True, preserve_range=True).astype(np.uint8)
            save(i, "%s_%d%s" % (filename_no_ext, sz, ext), storage_dir, local_filename, local_dir)


def get_silhouette_mask(img, config):
    """
    Using the watershed algorithm, generates a segmentation mask separating the foreground of an image from the white background.

    :param img: The image to be segmented.
    :param config: The configuration map which can contain parameters for the segmentation.
    :return: The silhouette mask, as a numpy array of boolean values.
    """
    def filter_mask(mask, size_factor=config.get('silo_filter_size_cutoff', 0.002)):
        labels, _ = ndi.label(mask)
        sizes = np.bincount(labels.ravel())
        mask_sizes = sizes > mask.size * size_factor
        mask_sizes[0] = 0
        return mask_sizes[labels]

    w, h, _ = img.shape
    gray = rgb2gray(img)

    # gets the edges
    sobel = filters.sobel(gray)
    blurred = filters.gaussian(sobel, sigma=2.0)

    # selects pure white areas that are a specified portion of the image
    white_areas = (gray >= config.get('silo_white_min_cutoff', 1.0))
    white_areas_filtered = filter_mask(white_areas)
    seed_mask = np.copy(white_areas_filtered) * 1

    # selects any pixels that are far enough from pure white
    gray_cutoff = config.get('silo_gray_max_cutoff', 0.95)
    min_img_not_background = config.get('silo_min_img_not_background', 0.1)
    while True:
        non_white_areas = (gray <= gray_cutoff)
        non_white_areas_filtered = filter_mask(non_white_areas)

        if np.sum(non_white_areas_filtered) > min_img_not_background * non_white_areas_filtered.size:
            break
        gray_cutoff += 0.01

    seed_mask += non_white_areas_filtered * 2

    # runs the watershed algorithm to label pixels as background (1) or foreground(2)
    ws = morphology.watershed(blurred, seed_mask)

    return ws == 2


IDM_ATTRIBUTES_LOC = ParameterizedFileLoc(["gs://hd-personalization-prod-data/MergedFullFeed/",
                                           ParameterizedFileLoc.DATE,
                                           "/ItemAttributes/*"])

PACKAGE_IMG_JSON_ELEM = "package_image"

PACKAGE_DETAILS_FILENAME = "packageDetails"
LAYER_IMGS_SUFFIX = "layer_imgs"
PACKAGE_IMGS_SUFFIX = "package_imgs"
PACKAGE_TO_IMAGE_ID_MAP_FILENAME = "package_to_image_mapping.csv"

PRIMARY_IMAGE_ATTRIBUTE_GUID = "645296e8-a910-43c3-803c-b51d3f1d4a89"
idm_attributes_file = "gs://hd-personalization-prod-data/MergedFullFeed/2020-10-20/ItemAttributes/*"