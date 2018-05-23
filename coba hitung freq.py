def yieldContext(l):
    l = [''] + l + ['']
    for i in enumerate(l):
        yield ' '.join(l[i-1:i+2]).strip()

def main():
    t1 = lambda x: x.startswith('')
    ex_paragraph = ['The quick brown fox jumps over the fence.', 
                   'Where there is another red fox.', 
                   'They run off together.', 
                   'They live hapily ever after.']
    print (list(yieldContext(ex_paragraph)))

main()