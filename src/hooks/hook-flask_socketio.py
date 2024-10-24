from PyInstaller.utils.hooks import collect_all, collect_submodules

# Include all DNS-related modules that eventlet needs
hiddenimports = [
    "engineio.async_drivers.eventlet",
    "eventlet.hubs",
    "rq",
    "redis",
    "flask_socketio",
    # DNS modules that eventlet.greendns needs
    "dns",
    "dns.dnssec",
    "dns.e164",
    "dns.edns",
    "dns.entropy",
    "dns.exception",
    "dns.flags",
    "dns.immutable",
    "dns.inet",
    "dns.ipv4",
    "dns.ipv6",
    "dns.message",
    "dns.name",
    "dns.namedict",
    "dns.node",
    "dns.opcode",
    "dns.query",
    "dns.rcode",
    "dns.rdata",
    "dns.rdataclass",
    "dns.rdataset",
    "dns.rdatatype",
    "dns.renderer",
    "dns.resolver",
    "dns.reversename",
    "dns.rrset",
    "dns.set",
    "dns.tokenizer",
    "dns.tsig",
    "dns.tsigkeyring",
    "dns.ttl",
    "dns.version",
    "dns.wire",
    "dns.zone",
]

# Collect additional submodules
hiddenimports.extend(collect_submodules("eventlet"))
hiddenimports.extend(collect_submodules("redis"))
hiddenimports.extend(collect_submodules("dns"))

# Collect binaries and data files
datas, binaries, _ = collect_all("eventlet")
