from colorama import Fore


def info(text, *args, **kwargs):
    print(Fore.GREEN + str(text) + Fore.RESET, *args, **kwargs)


def INFO(text, *args, **kwargs):
    print(Fore.GREEN + "[GENOME] INFO: " + str(text) + Fore.RESET, *args, **kwargs)


def warn(text, *args, **kwargs):
    print(Fore.YELLOW + str(text) + Fore.RESET, *args, **kwargs)


def WARN(text, *args, **kwargs):
    print(Fore.YELLOW + "[GENOME] WARNING: " + str(text) + Fore.RESET, *args, **kwargs)


def critical(text, *args, **kwargs):
    print(Fore.RED + str(text) + Fore.RESET, *args, **kwargs)


def CRITICAL(text, *args, **kwargs):
    print(Fore.RED + "[GENOME] CRITICAL: " + str(text) + Fore.RESET, *args, **kwargs)
