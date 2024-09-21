#ifndef BIGNUM_HPP
#define BIGNUM_HPP

#include <algorithm> 
#include <random>
#include <bitset>
#include <iostream>
#include <iomanip>
#include <chrono> 
#include <tuple> 
#include <cstdlib>  
#include <ctime> 
#include <string> 
#include <cmath>
#include <vector>
#define BASE (uint64_t(1) << 32)

class Bignum {
public:

  void setPositive(bool);
  void printHex() const;

  Bignum(int = 0);

  Bignum(std::string s);

  Bignum(Bignum const&) = default;
  Bignum(Bignum&&) = default;

  Bignum& operator=(Bignum const&) = default;
  Bignum& operator=(Bignum&&) = default;

  Bignum& operator+=(Bignum const& x) { *this = *this + x; return *this; };
  Bignum& operator++(int) { *this = (*this + 1);return *this; }

  Bignum& operator-=(Bignum const& x) { *this = *this - x; return *this; };
  Bignum& operator--(int) { *this = (*this - 1);return *this; }

  Bignum& operator*=(Bignum const& x) { *this = *this * x; return *this; };
  Bignum& operator/=(Bignum const& x) { *this = *this / x; return *this; };
  Bignum& operator%=(Bignum const& x) { *this = (*this % x); return *this; }

  Bignum& operator<<=(unsigned long const&);
  Bignum& operator>>=(unsigned long const&);

  uint32_t& operator[](unsigned i) { return tab[i]; };
  uint32_t operator[](unsigned i) const { return tab[i]; };
  size_t size() const { return tab.size(); };

private:

  std::vector<uint32_t> tab;
  bool isPositive;

  friend bool operator<(Bignum const&, Bignum const&);
  friend bool operator<=(Bignum const&, Bignum const&);
  friend bool operator>(Bignum const&, Bignum const&);
  friend bool operator>=(Bignum const&, Bignum const&);
  friend bool operator==(Bignum const&, Bignum const&);
  friend bool operator!=(Bignum const&, Bignum const&);

  friend std::ostream& operator<<(std::ostream&, Bignum const&);

  friend Bignum operator+(Bignum const&, Bignum const&);
  friend Bignum operator-(Bignum const&, Bignum const&);
  friend Bignum operator*(Bignum const&, Bignum const&);

  friend Bignum operator/(Bignum const&, Bignum const&);
  friend Bignum operator%(Bignum const&, Bignum const&);

  friend Bignum operator<<(Bignum const&, unsigned long const&);
  friend Bignum operator>>(Bignum const&, unsigned long const&);
  friend std::string divisePar2(std::string s);

  friend std::pair<Bignum, Bignum> division(Bignum const&, Bignum const&);

  friend Bignum mod_pow(const Bignum& base, const Bignum& exponent, const Bignum& modulo);

  friend Bignum generateRandomBignum(const Bignum& n);
  friend Bignum generateRandomBignumRange(const Bignum& n);

  friend Bignum getRandomPrime(Bignum const&);
  friend bool millerRabinTest(const Bignum& n, int k);

  friend std::tuple<Bignum, Bignum, Bignum> EEA(Bignum a, Bignum b);
  friend Bignum modularInverse(Bignum const&, Bignum const&);
  friend std::pair<std::pair<Bignum, Bignum>, std::pair<Bignum, Bignum>> generateKeys(int keysize);

  friend std::vector<Bignum> encodeData(const std::string& input, int blockSize);
  friend std::string decodeData(std::vector<Bignum> encoded);

  friend std::vector<Bignum>  encrypt(std::vector<Bignum> data, Bignum n_privateKey, Bignum d_privateKey);
  friend std::vector<Bignum>  decrypt(std::vector<Bignum>  cypher, Bignum n_publicKey, Bignum e_publicKey);

};

#endif
