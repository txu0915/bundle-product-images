


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
