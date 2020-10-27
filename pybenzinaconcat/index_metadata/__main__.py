from os.path import basename, dirname
from pybenzinaconcat.utils.jugutils import jug_main


def _main():
    from pybenzinaconcat.index_metadata.index_metadata import main
    main()


jug_main(_main, __file__, basename(dirname(__file__)))
