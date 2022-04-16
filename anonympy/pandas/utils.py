from typing import List
from time import time
import pandas as pd


faker_methods = "A | aba, address, administrative_unit, am_pm, " \
                "android_platform_token, ascii_company_email, ascii_email, "\
                "ascii_free_email, ascii_safe_email\n" \
                "B | bank_country, bban, boolean, bothify, bs, "\
                "building_number\n"\
                "C | cache_pattern, catch_phrase, century, chrome, city, "\
                "city_prefix, "\
                "city_suffix, color, color_name, company, company_email, "\
                "company_suffix, "\
                "coordinate, country, country_calling_code, country_code, "\
                "credit_card_expire,"\
                "credit_card_full, credit_card_number, credit_card_provider, "\
                "credit_card_security_code, cryptocurrency, "\
                "cryptocurrency_code, "\
                "cryptocurrency_name, csv, currency, currency_code, "\
                "currency_name, "\
                "currency_symbol, current_country, current_country_code\n"\
                "D | date, date_between, date_between_dates, date_object, "\
                "date_of_birth, "\
                "date_this_century, date_this_decade, date_this_month, "\
                "date_this_year, "\
                "date_time, date_time_ad, date_time_between, "\
                "date_time_between_dates, "\
                "date_time_this_century, date_time_this_decade, "\
                "date_time_this_month, "\
                "date_time_this_year, day_of_month, day_of_week, "\
                "del_arguments, dga, "\
                "domain_name, domain_word, dsv\n"\
                "E | ean, ean13, ean8, ein, email\n"\
                "F | factories, file_extension, file_name,file_path,firefox, "\
                "first_name,"\
                "first_name_female, first_name_male,first_name_nonbinary,"\
                "fixed_width, format,"\
                "free_email, free_email_domain, future_date, "\
                "future_datetime\n"\
                "G | generator_attrs, get_arguments, get_formatter, "\
                "get_providers\n"\
                "H | hex_color, hexify, hostname, http_method\n"\
                "I | iana_id, iban, image, image_url, internet_explorer, "\
                "invalid_ssn, "\
                "ios_platform_token, ipv4, ipv4_network_class, ipv4_private, "\
                "ipv4_public, "\
                "ipv6, isbn10, isbn13, iso8601, items, itin\n"\
                "J | job, json\n"\
                "L | language_code,language_name,last_name,last_name_female, "\
                "last_name_male, "\
                "last_name_nonbinary, latitude, latlng, lexify, "\
                "license_plate, "\
                "linux_platform_token, linux_processor, local_latlng, locale,"\
                "locales, "\
                "localized_ean,localized_ean13, localized_ean8, "\
                "location_on_land, longitude\n"\
                "M | mac_address, mac_platform_token, mac_processor, md5, "\
                "military_apo, "\
                "military_dpo, military_ship, military_state, mime_type, "\
                "month, month_name,"\
                " msisdn\n"\
                "N | name, name_female, name_male, name_nonbinary, "\
                "nic_handle, nic_handles, "\
                "null_boolean, numerify\n"\
                "O | opera\n"\
                "P | paragraph, paragraphs, parse, password, past_date, "\
                "past_datetime, "\
                "phone_number, port_number, postalcode, postalcode_in_state, "\
                "postalcode_plus4, postcode, postcode_in_state, prefix, "\
                "prefix_female, prefix_male, prefix_nonbinary, pricetag, "\
                "profile, "\
                "provider, providers, psv, pybool, pydecimal, pydict, "\
                "pyfloat, pyint,"\
                "pyiterable, pylist, pyset, pystr, pystr_format, pystruct, "\
                "pytimezone, "\
                "pytuple\n"\
                "R | random, random_choices, random_digit, "\
                "random_digit_not_null, "\
                "random_digit_not_null_or_empty, random_digit_or_empty, "\
                "random_element, "\
                "random_elements, random_int, random_letter, random_letters, "\
                "random_lowercase_letter, random_number, random_sample, "\
                "random_uppercase_letter, randomize_nb_elements, rgb_color, "\
                "rgb_css_color, ripe_id\n"\
                "S | safari, safe_color_name, safe_domain_name, safe_email, "\
                "safe_hex_color, "\
                "secondary_address, seed_instance, seed_locale, sentence, "\
                "sentences, "\
                "set_arguments, set_formatter, sha1, sha256, simple_profile, "\
                "slug, ssn, "\
                "state, state_abbr, street_address, street_name, "\
                "street_suffix, suffix, "\
                "suffix_female, suffix_male, suffix_nonbinary, swift, "\
                "swift11, swift8"\
                "T | tar, text, texts, time, time_delta, time_object, "\
                "time_series, timezone, "\
                "tld, tsv\n"\
                "U | unique, unix_device, unix_partition, unix_time, "\
                "upc_a, upc_e, uri, "\
                "uri_extension, uri_page, uri_path, url, user_agent, "\
                "user_name, uuid4\n"\
                "W | weights, windows_platform_token, word, words\n"\
                "Y | year\n"\
                "Z | zipcode, zipcode_in_state, zipcode_plus4"

_fake_methods = ['aba', 'address', 'administrative_unit', 'am_pm',
                 'android_platform_token',
                 'ascii_company_email', 'ascii_email', 'ascii_free_email',
                 'ascii_safe_email', 'bank_country', 'bban',
                 'boolean', 'bothify', 'bs', 'building_number', 'catch_phrase',
                 'century',
                 'chrome', 'city', 'city_prefix', 'city_suffix', 'color',
                 'color_name', 'company', 'company_email',
                 'company_suffix', 'coordinate', 'country',
                 'country_calling_code', 'country_code', 'credit_card_expire',
                 'credit_card_full', 'credit_card_number',
                 'credit_card_provider',
                 'credit_card_security_code', 'cryptocurrency',
                 'cryptocurrency_code', 'cryptocurrency_name',
                 'csv', 'currency',
                 'currency_code', 'currency_name', 'currency_symbol',
                 'current_country', 'current_country_code', 'date',
                 'date_between', 'date_between_dates', 'date_object',
                 'date_of_birth',
                 'date_this_century', 'date_this_decade', 'date_this_month',
                 'date_this_year', 'date_time', 'date_time_ad',
                 'date_time_between',
                 'date_time_between_dates', 'date_time_this_century',
                 'date_time_this_decade', 'date_time_this_month',
                 'date_time_this_year',
                 'day_of_month', 'day_of_week', 'del_arguments', 'dga',
                 'domain_name', 'domain_word', 'dsv', 'ean', 'ean13', 'ean8',
                 'ein', 'email',
                 'factories', 'file_extension', 'file_name', 'file_path',
                 'firefox', 'first_name', 'first_name_female',
                 'first_name_male', 'first_name_nonbinary',
                 'fixed_width', 'format', 'free_email',
                 'free_email_domain', 'future_date', 'future_datetime',
                 'generator_attrs',
                 'get_arguments', 'get_formatter', 'get_providers',
                 'hex_color', 'hexify', 'hostname', 'http_method', 'iana_id',
                 'iban', 'image', 'image_url', 'internet_explorer',
                 'invalid_ssn', 'ios_platform_token',
                 'ipv4', 'ipv4_network_class', 'ipv4_private', 'ipv4_public',
                 'ipv6', 'isbn10', 'isbn13',
                 'iso8601', 'items', 'itin', 'job', 'json', 'language_code',
                 'language_name', 'last_name', 'last_name_female',
                 'last_name_male', 'last_name_nonbinary',
                 'latitude', 'latlng', 'lexify', 'license_plate',
                 'linux_platform_token', 'linux_processor', 'local_latlng',
                 'locale', 'locales', 'localized_ean', 'localized_ean13',
                 'localized_ean8', 'location_on_land', 'longitude',
                 'mac_address', 'mac_platform_token', 'mac_processor', 'md5',
                 'military_apo', 'military_dpo', 'military_ship',
                 'military_state', 'mime_type', 'month',
                 'month_name', 'msisdn',
                 'name', 'name_female', 'name_male', 'name_nonbinary',
                 'nic_handle', 'nic_handles', 'null_boolean',
                 'numerify', 'opera', 'paragraph', 'paragraphs', 'parse',
                 'password', 'past_date', 'past_datetime', 'phone_number',
                 'port_number', 'postalcode', 'postalcode_in_state',
                 'postalcode_plus4', 'postcode', 'postcode_in_state', 'prefix',
                 'prefix_female', 'prefix_male', 'prefix_nonbinary',
                 'pricetag', 'profile', 'provider', 'providers', 'psv',
                 'pybool', 'pydecimal', 'pydict', 'pyfloat', 'pyint',
                 'pyiterable', 'pylist', 'pyset', 'pystr', 'pystr_format',
                 'pystruct', 'pytimezone', 'pytuple',
                 'random', 'random_choices', 'random_digit',
                 'random_digit_not_null', 'random_digit_not_null_or_empty',
                 'random_digit_or_empty',
                 'random_element', 'random_elements', 'random_int',
                 'random_letter', 'random_letters',
                 'random_lowercase_letter', 'random_number', 'random_sample',
                 'random_uppercase_letter', 'randomize_nb_elements',
                 'rgb_color', 'rgb_css_color', 'ripe_id', 'safari',
                 'safe_color_name',
                 'safe_domain_name', 'safe_email', 'safe_hex_color',
                 'secondary_address', 'seed', 'seed_instance', 'seed_locale',
                 'sentence', 'sentences', 'set_arguments', 'set_formatter',
                 'sha1', 'sha256', 'simple_profile',
                 'slug', 'ssn', 'state', 'state_abbr', 'street_address',
                 'street_name', 'street_suffix', 'suffix', 'suffix_female',
                 'suffix_male',
                 'suffix_nonbinary', 'swift', 'swift11', 'swift8', 'tar',
                 'text', 'texts', 'time', 'time_delta', 'time_object',
                 'time_series', 'timezone', 'tld', 'tsv', 'unique',
                 'unix_device', 'unix_partition', 'unix_time', 'upc_a',
                 'upc_e', 'uri', 'uri_extension', 'uri_page', 'uri_path',
                 'url',
                 'user_agent', 'user_name', 'uuid4', 'weights',
                 'windows_platform_token', 'word', 'words', 'year', 'zipcode',
                 'zipcode_in_state', 'zipcode_plus4']

av_methods = '''`numeric`:
        * Perturbation - "numeric_noise"
        * Binning - "numeric_binning"
        * PCA Masking - "numeric_masking"
        * Rounding - "numeric_rounding"

`categorical`:
        * Synthetic Data - "categorical_fake"
        * Synthetic Data Auto - "categorical_fake_auto"
        * Resampling from same Distribution - "categorical_resampling"
        * Tokenazation - "categorical_tokenization"
        * Email Masking - "categorical_email_masking"

`datetime`:
        * Synthetic Date - "datetime_fake"
        * Perturbation - "datetime_noise"

`general`:
        * Drop Column - "column_suppression"
        '''


def get_numeric_columns(df) -> List:
    '''
    Return a subset of the DataFrame's columns which are numeric.

    Returns
    ----------
        List of columns with numeric values
    '''
    return df.select_dtypes('number').columns.tolist()


def get_categorical_columns(df) -> List:
    '''
    Return a subset of the DataFrame's columns which are categorical.

    Returns
    ----------
        List of columns with categorical values
    '''
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def get_datetime_columns(df) -> List:
    '''
    Return a subset of the DataFrame's columns which are of datetime type.

    Returns
    ----------
        List of columns with datetime values
    '''
    return df.select_dtypes(include=['datetime']).columns.tolist()


def timer_func(func):
    '''
    This decorator-function shows the execution time of the function
    '''
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


def fake_methods(letter=None):
    '''
    Print a list of faker's methods
    '''
    if (letter is None) or (letter == 'all'):
        print(faker_methods)
    else:
        for line in faker_methods.split('\n'):
            if line[0] == letter.upper():
                print(line[3:])


def available_methods(dtype=None):
    '''
    Print a list of available methods
    '''
    splitted = {'numeric': ['numeric_noise', 'numeric_binning',
                            'numeric_masking', 'numeric_rounding'],
                'categorical': ['categorical_fake', 'categorical_fake_auto',
                                'categorical_resampling',
                                'categorical_tokenization',
                                'categorical_email_masking'],
                'datetime': ["datetime_fake", "datetime_noise"],
                'general': ['column_suppression']}
    if (dtype is None) or (dtype == 'all'):
        print(av_methods)
    else:
        print(*splitted[dtype], sep='\t')


def load_dataset(size='small'):
    '''
    Sample dataset for demonstration purposes only

    Returns
    ----------
        df : pd.DataFrame
    '''
    if size == 'small':
        df = pd.DataFrame({
            "name": ["Bruce", "Tony"],
            "age": [33, 48],
            "birthdate": [pd.Timestamp(1915, 4, 17),
                          pd.Timestamp(1970, 5, 29)],
            "salary": [59234.32, 49324.53],
            "web": ['http://www.alandrosenburgcpapc.co.uk',
                    'http://www.capgeminiamerica.co.uk'],
            "email": ['josefrazier@owen.com', 'eryan@lewis.com'],
            "ssn": ["343554334", "656564664"]})

        return df
    elif size == 'big':
        df = pd.read_csv('https://raw.githubusercontent.com/ArtLabss/'
                         'open-data-anonimizer/main/examples/files/new.csv',
                         parse_dates=['birthdate'])
        return df
    else:
        raise ValueError("size takes only two values: 'small' / 'big' ")
