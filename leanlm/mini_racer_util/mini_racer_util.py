if __name__ == "__main__":
    from py_mini_racer import MiniRacer
    ctx = MiniRacer()
    print(ctx.eval("1+1"))
    print(ctx.eval("var x = {company: 'Sqreen'}; x.company"))
    print(ctx.eval("'❤'"))
    print(ctx.eval("var fun = () => ({ foo: 1 });"))
    

