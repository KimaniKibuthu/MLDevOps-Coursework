'''
Two functions checking datatypes
'''
import logging

# Setup logging configuration
logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def divide_vals(numerator, denominator):
    '''
    Args:
        numerator: (float) numerator of fraction
        denominator: (float) denominator of fraction

    Returns:
        fraction_val: (float) numerator/denominator
    '''
    try:
        # Assert the value types
        logging.info('%s, %s', numerator, denominator)
        #assert isinstance(numerator, int)
        #assert isinstance(denominator, int)
        
        # Get the fraction
        fraction_val = numerator/denominator
        logging.info('SUCCESS: The fraction is true')
        return fraction_val
    
    except ZeroDivisionError:
        logging.error('ERROR: Denominator cannot be zero')
        
        
def num_words(text):
    '''
    Args:
        text: (string) string of words

    Returns:
        num_words: (int) number of words in the string
    '''
    try:
        logging.info('%s', text)
        #assert isinstance(text, str)
        num_words = len(text.split())
        logging.info(f'SUCCESS: The length of the string is {num_words}')
        return num_words
    
    except AttributeError:
        logging.error('ERROR: text not string')

        
if __name__ == "__main__":
    divide_vals(3.4, 0)
    divide_vals(4.5, 2.7)
    divide_vals(-3.8, 2.1)
    divide_vals(1, 2)
    num_words(5)
    num_words('This is the best string')
    num_words('one')
