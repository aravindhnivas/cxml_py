import umdalib.server as server


class Args:
    port = 18704
    debug = False


args = Args()
print(f"port: {args.port}, debug: {args.debug}")
server.main(args)
