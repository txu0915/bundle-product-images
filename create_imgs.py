import sys
sys.path.append("/Users/tianlongxu/Documents/My Projects/Kitchen_packages/")
import pyspark
from py_scripts.utilities import *
from pyspark.sql import *
import findspark
import pandas as pd
os.environ["SPARK_HOME"] = "/Users/tianlongxu/anaconda3/lib/python3.8/site-packages/pyspark"  #"/Users/tianlongxu/Documents/My Projects/Kitchen_packages/venv/lib/python3.7/site-packages/pyspark"
os.environ["PYTONPATH"] = '/Users/tianlongxu/anaconda3/lib/python3.8/site-packages/pyspark/python/lib/py4j-0.10.9-src.zip'
# os.environ["PYTONPATH"] = "/Users/tianlongxu/Documents/My Projects/Kitchen_packages/venv/bin/python:/Users/tianlongxu/Documents/My Projects/Kitchen_packages/venv/lib/python3.7/site-packages/pyspark/python/lib/py4j-0.10.7-src.zip"
#findspark.init("/Users/tianlongxu/Downloads/spark-2.4.3-bin-hadoop2.7/")


package_dir = "gs://hd-datascience-np-data/kitchen-package"
layout_config_filepath = "gs://hd-datascience-np-artifacts/jim/kitchen_package_images_changed/config/bundle_layouts1.json"
image_config_filepath = "gs://hd-datascience-np-artifacts/jim/kitchen_package_images_changed/config/image_config.json"
img_dir = "gs://hd-datascience-np-data/kitchen-package"
input_filepath = 'gs://hd-datascience-np-artifacts/tianlong/kitchen_packages/Inputs/tools-package-test-2.txt'
#input_filepath = 'gs://hd-datascience-np-artifacts/tianlong/kitchen_packages/Inputs/kitchen_package_details_LaborDay_forImage.txt'
img_dir_layer = 'gs://hd-datascience-np-artifacts/tianlong/kitchen_packages/Outputs/layer_imgs_test/'
img_dir_package = 'gs://hd-datascience-np-artifacts/tianlong/kitchen_packages/Outputs/package_imgs_test/'

day = get_most_recent_date(package_dir)

day = '2020-10-20'
spark = SparkSession.builder.master('local[*]').getOrCreate()
#spark = SparkSession.builder.appName('GCSFilesRead').getOrCreate()

layout_config = get_config("./Configs/bundle_layouts.json")
image_config = get_config("./Configs/image_config.json")
kitchen_package_layout = get_layout(layout_config, 'tools')

product_list_df = get_product_list_df(input_filepath, day, spark)
layer_skus_df = get_layer_skus_df(product_list_df)
product_list_df = filter_product_list_df(product_list_df)


#day = datetime.datetime.today().strftime('%Y-%m-%d')
idm_attributes_file = "./Data/itemattributes-tools.csv"
sku_img_df = get_sku_img_df(layer_skus_df, day, spark, idm_attributes_file)
layer_img_folder = "./layer_img_folder/"
# line_key, img_id, sku_list
all_product_sets_df = get_all_product_sets_df(product_list_df, kitchen_package_layout)

layer_img_folder = create_and_save_layer_imgs(sku_img_df, kitchen_package_layout, image_config, img_dir,local_layer_folder_name="layer_img_folder")

create_and_save_package_imgs(all_product_sets_df, layer_img_folder, image_config, img_dir_package)




all_skus = pd.read_csv('./Data/19-cat-skus.csv')
bq_attributes = pd.read_csv('./Data/itemattributes-tools.csv')
itc_comp = pd.read_csv('Data/complementary-results-from-itc.csv')
for i,sku in all_skus.iterrows():
    if sku.sku not in bq_attributes.oms_id.values:
        print(str(sku.sku).split('.')[0])

sku_dic = {}

for i in [310179064,309677412,309415135,310782220,311594940,312370596,311595564,312462409,312306957,311454336,312493945]:
    if i not in sku_dic:
        sku_dic[i] = 1
    else:
        sku_dic[i] += 1
