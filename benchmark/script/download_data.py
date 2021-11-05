import sys
import os
import urllib


def main():
    data_type = sys.argv[1]
    conf_filepath = "../conf/%s_dataset_urls" % data_type
    conf_file = open(conf_filepath)
    for line in conf_file:
        (dataset_name, url) = line.split()
        print dataset_name
        dataset_path = "../data/%s.bz2" % dataset_name
        urllib.urlretrieve(url, dataset_path)
        os.system("bunzip2 " + dataset_path)

    conf_file.close()


if __name__ == "__main__":
    main()
