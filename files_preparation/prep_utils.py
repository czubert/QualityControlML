import re



def pattern_in_name(name, re_pattern):
    if re.search(re_pattern, name) is None:
        return False
    elif re.search(re_pattern, name) is not None:
        return True

