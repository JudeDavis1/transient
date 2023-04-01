from colorama import Fore


def info(text, *args, **kwargs):
    print(Fore.GREEN + str(text) + Fore.RESET, *args, **kwargs)


def special(text, *args, **kwargs):
    print(Fore.BLUE + str(text) + Fore.RESET, *args, **kwargs)


def SPECIAL(text, *args, **kwargs):
    print(Fore.GREEN + "[TRANSIENT] INFO: " + str(text) + Fore.RESET, *args, **kwargs)


def INFO(text, *args, **kwargs):
    print(Fore.GREEN + "[TRANSIENT] INFO: " + str(text) + Fore.RESET, *args, **kwargs)


def warn(text, *args, **kwargs):
    print(Fore.YELLOW + str(text) + Fore.RESET, *args, **kwargs)


def WARN(text, *args, **kwargs):
    print(Fore.YELLOW + "[TRANSIENT] WARNING: " + str(text) + Fore.RESET, *args, **kwargs)


def critical(text, *args, **kwargs):
    print(Fore.RED + str(text) + Fore.RESET, *args, **kwargs)


def CRITICAL(text, *args, **kwargs):
    print(Fore.RED + "[TRANSIENT] CRITICAL: " + str(text) + Fore.RESET, *args, **kwargs)
