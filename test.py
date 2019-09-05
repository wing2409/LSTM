from ner import Parser

p = Parser()
p.load_models("models/")

print(p.predict("한편, AFC챔피언스리그 E조에 속한 포항 역시 대회 8강 진출이 불투명하다"))
print(p.predict("2003년 6월 14일 사직 두산전 이후 박명환에게 당했던 10연패 사슬을 거의 5년 만에 끊는 의미있는 승리였다"))
print(p.predict("선덜랜드(원정) 등 약체들과 대진을 남겨 놓고 있다"))
print(p.predict("원주 동부 관중이 없어요"))
print(p.predict("롯데 4점 정도를 예상한다"))
print(p.predict("SK가 롯데에게 승리를 거뒀다"))
print(p.predict("박찬호는 대한민국 야구선수다"))
print(p.predict("결국 박찬호는 버크에게 초구 94마일 패스트볼로 헛스윙을 유도한 뒤 87마일 슬라이더로 2루 플라이를 솎아내며 거뜬히 위기를 넘겼습니다"))
print(p.predict("박항서 전남 드래곤즈 감독 -이번 경기에 임하는 각오는"))

