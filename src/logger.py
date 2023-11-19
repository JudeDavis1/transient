from colorama import Fore


def info(text, *args, engine=print, **kwargs):
    engine(Fore.GREEN + str(text) + Fore.RESET, *args, **kwargs)


def special(text, *args, engine=print, **kwargs):
    engine(Fore.BLUE + str(text) + Fore.RESET, *args, **kwargs)


def SPECIAL(text, *args, engine=print, **kwargs):
    engine(Fore.GREEN + "[TRANSIENT] INFO: " + str(text) + Fore.RESET, *args, **kwargs)


def INFO(text, *args, engine=print, **kwargs):
    engine(Fore.GREEN + "[TRANSIENT] INFO: " + str(text) + Fore.RESET, *args, **kwargs)


def warn(text, *args, engine=print, **kwargs):
    engine(Fore.YELLOW + str(text) + Fore.RESET, *args, **kwargs)


def WARN(text, *args, engine=print, **kwargs):
    engine(
        Fore.YELLOW + "[TRANSIENT] WARNING: " + str(text) + Fore.RESET, *args, **kwargs
    )


def critical(text, *args, engine=print, **kwargs):
    engine(Fore.RED + str(text) + Fore.RESET, *args, **kwargs)


def CRITICAL(text, *args, engine=print, **kwargs):
    engine(
        Fore.RED + "[TRANSIENT] CRITICAL: " + str(text) + Fore.RESET, *args, **kwargs
    )
