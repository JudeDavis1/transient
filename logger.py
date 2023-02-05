from colorama import Fore


def info(text, *args, **kwargs):
    print(Fore.GREEN + text + Fore.RESET, *args, **kwargs)

def INFO(text, *args, **kwargs):
    print(Fore.GREEN + "[GENOME] INFO: " + text + Fore.RESET, *args, **kwargs)

def warn(text, *args, **kwargs):
    print(Fore.YELLOW + text + Fore.RESET, *args, **kwargs)

def WARN(text, *args, **kwargs):
    print(Fore.YELLOW + "[GENOME] WARNING: " + text + Fore.RESET, *args, **kwargs)

def critical(text, *args, **kwargs):
    print(Fore.RED + text + Fore.RESET, *args, **kwargs)

def CRITICAL(text, *args, **kwargs):
    print(Fore.RED + "[GENOME] CRITICAL: " + text + Fore.RESET, *args, **kwargs)
