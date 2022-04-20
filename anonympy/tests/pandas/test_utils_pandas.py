import urllib
import pytest
from anonympy.pandas import dfAnonymizer
from anonympy.pandas.utils_pandas import load_dataset, \
     fake_methods, available_methods


# testing utils.py - supplementary functions


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


@pytest.fixture(scope="module", params=["small", "big"])
def load_df(request):
    '''
    Returns a DataFrame from anonympy.pandas.utils.load_dataset('small')
    '''
    try:
        df = load_dataset(request.param)
        return df
    except urllib.error.HTTPError:
        df = None
        return df


@pytest.fixture(scope="module")
def anonym_obj(load_df):
    '''
    Returns dfAnonymizer instance
    '''
    if load_df is not None:
        anonym = dfAnonymizer(load_df)
        return anonym


@pytest.fixture(scope="module")
def anonym_small():
    df = load_dataset('small')
    anonym = dfAnonymizer(df)
    return anonym


@pytest.fixture(scope="module")
def anonym_big():
    try:
        df = load_dataset('big')
        anonym = dfAnonymizer(df)
        return anonym
    except urllib.error.HTTPError:
        anonym = None
        return anonym


def test_load_dataset(load_df):
    if load_df is None:
        assert False, "`load_dataset('big')` should have returned a pandas \
        DataFrame, which is fetched from https://github.com/ArtLabss/\
        open-data-anonymizer/blob/main/examples/files/new.csv"
    else:
        assert True


def test_fake_methods(capsys):
    for i in range(ord('a'), ord('z')+1):
        letter = chr(i)
        fake_methods(letter)
        captured = capsys.readouterr()
        for line in faker_methods.split('\n'):
            if line[0] == letter.upper():
                expected = line[3:]
                assert (captured.out).strip() == expected.strip(), f"`fake\
                _methods()` returned unexpected methods \
                for letter - `{letter.upper()}`"


def test_available_methods(capsys):
    splitted = {'numeric': ['numeric_noise',
                            'numeric_binning',
                            'numeric_masking',
                            'numeric_rounding'],
                'categorical': ['categorical_fake',
                                'categorical_fake_auto',
                                'categorical_resampling',
                                'categorical_tokenization',
                                'categorical_email_masking'],
                'datetime': ["datetime_fake",
                             "datetime_noise"],
                'general': ['column_suppression']}
    for dtype in splitted.keys():
        available_methods(dtype)
        captured = capsys.readouterr()
        assert (captured.out).strip() == '\t'.join(splitted[dtype]), f"`avail\
        able_methods()` returned unexpected methods for dtype - `{dtype}`"


def test_get_numeric_columns_small(anonym_small):
    assert anonym_small.numeric_columns == ['age', 'salary'], "Fucntion\
     `get_numeric_columns` from anonympy.pandas.utils returned\
      unexpected columns for `load_dataset('small')`"


def test_get_numeric_columns_big(anonym_big):
    if anonym_big is None:
        pytest.skip("Failed to fetch the DataFrame")
    assert anonym_big.numeric_columns == ['salary', 'age'], "Fucntion \
    `get_numeric_columns` from anonympy.pandas.utils \
    returned unexpected columns for `load_dataset('big')`"


def test_get_categorical_columns_small(anonym_small):
    assert anonym_small.categorical_columns == ['name', 'web', 'email', 'ssn'], "Fucntion `get_categorical_columns` from anonympy.pandas.utils returned unexpected columns for `load_dataset('small')`"  # noqa: E501


def test_get_categorical_columns_big(anonym_big):
    if anonym_big is None:
        pytest.skip("Failed to fetch the DataFrame")
    assert anonym_big.categorical_columns == ['first_name',
                                              'address', 'city',
                                              'phone', 'email', 'web',
                                              'birthdate'], "Fucntion\
                                               `get_categorical_columns` \
                                               from anonympy.pandas.utils \
                                               returned unexpected columns \
                                               for `load_dataset('big')`"


def test_get_datetime_columns_small(anonym_small):
    assert anonym_small.datetime_columns == ['birthdate'], "Fucntion \
    `get_datetime_columns` from anonympy.pandas.utils \
    returned unexpected columns for `load_dataset('small')`"


def test_get_datetime_columns_big(anonym_big):
    if anonym_big is None:
        pytest.skip("Failed to fetch the DataFrame")
    assert anonym_big.datetime_columns == [], "Fucntion `get_datetime_columns`\
    from anonympy.pandas.utils returned unexpected columns for \
    `load_dataset('big')`"
