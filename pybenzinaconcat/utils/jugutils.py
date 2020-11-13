import os.path

_PKG = os.path.basename(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                     "..")))


def jug_main(func, jugfile, pkg=_PKG):
    from jug import is_jug_running
    if is_jug_running():
        func()
    else:
        import argparse
        from os.path import basename, dirname
        from jug.jug import main
        if pkg is None:
            pkg = basename(dirname(jugfile))
        p = argparse.ArgumentParser()
        p.add_argument("--jugdir", default="{}.jugdir/".format(pkg))
        args, argv = p.parse_known_args()
        main(["jug", "execute", "--jugdir", args.jugdir, jugfile] + argv)
