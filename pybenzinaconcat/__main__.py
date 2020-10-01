from jug import is_jug_running

if is_jug_running():
    from pybenzinaconcat import pybenzinaconcat, parse_args
    pybenzinaconcat(*parse_args())
else:
    from os.path import basename, dirname
    from sys import argv
    from jug.jug import main
    argv = ["jug", "execute",
            "--jugdir", "{}.jugdir/".format(basename(dirname(__file__))),
            "--", __file__] + argv[1:]
    main(argv)
