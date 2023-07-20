import re

def filter_emoji(desstr,restr=''):
  try:
       co = re.compile(u'[\U00010000-\U0010ffff]|[\U0001F300-\U0001F64F]|[\U0001F680-\U0001F6FF]|[\u2600-\u2B55]+|¤*|Ψ*')
  except re.error:
       co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
  return co.sub(restr, desstr)

s="精彩请移步👇 👇 🔞🔞🔞🈲🈲[鲜花][鲜花][鲜花]传送@魅惑-御姐院[微风]["
print(filter_emoji(s))