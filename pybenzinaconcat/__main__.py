from jug import is_jug_running

if is_jug_running():
    from pybenzinaconcat.benzinaconcat import main
    main()
else:
    import argparse
    from os.path import basename, dirname
    from jug.jug import main
    pkg = basename(dirname(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--jugdir", default="{}.jugdir/".format(pkg))
    args, argv = p.parse_known_args()
    main(["jug", "execute", "--jugdir", args.jugdir, __file__] + argv)
