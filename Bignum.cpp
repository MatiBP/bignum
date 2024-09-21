#include "Bignum.hpp"

using namespace std;

// Constructor for creating a Bignum from an integer value
Bignum::Bignum(int value) {
    if (value >= 0) {
        isPositive = true;
        tab.clear();
        tab.push_back(value);
    }
    else {
        isPositive = false;
        tab.clear();
        tab.push_back(-value);
    }
}

// Print the Bignum in hexadecimal format
void Bignum::printHex() const {
    if (!isPositive) cout << "-";

    bool first = true;

    for (auto rit = tab.rbegin(); rit != tab.rend(); ++rit) {
        // Skip leading zeros
        if (first && *rit == 0) continue;

        cout << hex;

        if (!first) cout << setfill('0') << setw(8);
        else first = false;

        cout << *rit << dec;
    }

    if (first) cout << 0;
}

// Function to divide a binary string by 2
// This function performs binary division by 2 on the input string.
// It iterates through the string, converting chunks into integers and updating the remainder.
std::string divisePar2(std::string s) {
    std::string res = "";
    unsigned int reste = 0;

    while (!s.empty()) {
        int x = s[0] - '0';
        reste = reste * 10 + x;

        if (reste >= 2) {
            int q = reste / 2;
            reste = reste % 2;
            res += std::to_string(q);
        }
        else {
            res += '0';
        }

        s = s.substr(1);
    }

    // Remove leading zeros
    if (res.empty()) {
        return "0";
    }

    while (res[0] == '0') {
        res = res.substr(1);
    }

    return res;
}

// Constructor for a Bignum class that converts a binary string to a Bignum, handling the sign and chunking into 32-bit integers.
Bignum::Bignum(std::string s) {
    if (s.empty()) {
        throw std::invalid_argument("the string is empty");
    }

    std::string s_bin = "";
    std::string sres = s;

    // Determine sign
    if (sres[0] == '-') {
        sres.erase(0, 1); // Remove the negative sign if present
        isPositive = false;
    }
    else {
        isPositive = true;
    }

    // Convert the input string to a binary string
    while (!sres.empty()) {
        char lastChar = sres.back();
        std::string lastCharStr(1, lastChar);
        int lastDigit = std::stoi(lastCharStr);

        // Convert odd digits to '1', even digits to '0'
        s_bin = s_bin + ((lastDigit % 2) == 0 ? '0' : '1');

        sres = divisePar2(sres); // Divide the input string by 2
    }

    // Divide the binary string into 32-bit chunks, convert to integers
    for (unsigned int i = 0; i < s_bin.size(); i += 32) {
        std::string s_bin32(s_bin, i, 32);
        std::reverse(s_bin32.begin(), s_bin32.end());
        uint64_t value = std::bitset<32>(s_bin32).to_ulong();
        tab.push_back(value);
    }
}


void Bignum::setPositive(bool isPositive) {
    this->isPositive = isPositive;
}

// Addition operator for Bignum
// Operator overload for addition of two Bignum objects.
Bignum operator+(const Bignum& num1, const Bignum& num2) {

    Bignum result;
    result.tab.clear();

    if (!(num1.isPositive == num2.isPositive)) {
        Bignum bisNum1 = num1;
        Bignum bisNum2 = num2;
        bisNum1.isPositive = true;
        bisNum2.isPositive = true;

        // Subtract smaller absolute value from larger absolute value
        if (bisNum1 > bisNum2) {
            result = bisNum1 - bisNum2;
            result.isPositive = num1.isPositive;
            return result;
        }
        else {
            result = bisNum2 - bisNum1;
            result.isPositive = num2.isPositive;
            return result;
        }
    }

    result.isPositive = num1.isPositive;

    Bignum largerNum;
    Bignum smallerNum;

    if (num1.tab.size() >= num2.tab.size()) {
        largerNum = num1;
        smallerNum = num2;
    }
    else {
        largerNum = num2;
        smallerNum = num1;
    }

    size_t minSize = std::min(smallerNum.tab.size(), largerNum.tab.size());
    size_t maxSize = std::max(smallerNum.tab.size(), largerNum.tab.size());

    uint64_t carry = 0;
    // Perform addition digit by digit
    for (size_t i = 0; i < minSize; i++) {
        uint64_t sum = carry + static_cast<uint64_t>(smallerNum.tab[i]) + static_cast<uint64_t>(largerNum.tab[i]);
        result.tab.push_back(sum % BASE);
        carry = sum / BASE;
    }

    // Add remaining digits of the larger number
    for (size_t i = minSize; i < maxSize; i++) {
        uint64_t sum = static_cast<uint64_t>(largerNum.tab[i]) + carry;
        result.tab.push_back(sum % BASE);
        carry = sum / BASE;
    }

    // If there is a carry after the loop, add it to the result
    if (carry > 0) {
        result.tab.push_back(carry);
    }

    return result;
}

// Equality operator for Bignum
bool operator==(Bignum const& b1, Bignum const& b2) {

    if (b1.isPositive != b2.isPositive) {
        return false;
    }

    if (b1.tab.size() != b2.tab.size()) {
        return false;
    }

    // Compare each digit of the Bignum
    for (size_t i = 0; i < b1.tab.size(); i++) {
        if (b1.tab[i] != b2.tab[i]) {
            return false;
        }
    }

    return true;
}

// Inequality operator for Bignum
// True if the Bignum objects are not equal, otherwise false.
bool operator!=(Bignum const& x, Bignum const& y) {
    return !(x == y);
}

// Greater-than operator for Bignum
bool operator>(const Bignum& b1, const Bignum& b2) {

    // Check signs
    if (b1.isPositive && !b2.isPositive) {
        return true;
    }
    else if (!b1.isPositive && b2.isPositive) {
        return false;
    }
    else if (!b1.isPositive && !b2.isPositive) {
        // Both numbers are negative, compare absolute values
        Bignum absB1 = b1;
        Bignum absB2 = b2;
        absB1.isPositive = true;
        absB2.isPositive = true;
        return absB2 > absB1;
    }

    // Both numbers are non-negative
    if (b1 == 0 && b2 == 0) {
        return false;
    }

    // Compare sizes of the underlying arrays
    if (b1.tab.size() > b2.tab.size()) {
        return b1.isPositive;
    }
    else if (b1.tab.size() < b2.tab.size()) {
        return !b1.isPositive;
    }

    // Compare digits starting from the most significant digit
    for (int i = b1.tab.size() - 1; i >= 0; --i) {
        if (b1.tab[i] > b2.tab[i]) {
            return b1.isPositive;
        }
        else if (b1.tab[i] < b2.tab[i]) {
            return !b1.isPositive;
        }
    }

    return false;
}

// Less-than-or-equal-to operator for Bignum
// True if x is less than or equal to y, otherwise false.
bool operator<=(Bignum const& x, Bignum const& y) {
    return !(x > y);
}

// Less-than operator for Bignum
// True if x is less than y, otherwise false.
bool operator<(Bignum const& x, Bignum const& y) {
    return (x <= y) && (x != y);
}

// Greater-than-or-equal-to operator for Bignum
// True if x is greater than or equal to y, otherwise false.
bool operator>=(Bignum const& x, Bignum const& y) {
    return !(x < y);
}

// Subtraction operator for Bignum
Bignum operator-(const Bignum& num1, const Bignum& num2) {

    Bignum result = 0;

    // Handle different signs
    if (!(num1.isPositive == num2.isPositive)) {
        Bignum positiveNum1 = num1;
        Bignum positiveNum2 = num2;
        positiveNum1.isPositive = true;
        positiveNum2.isPositive = true;

        // Subtract absolute values and assign the correct sign
        result = positiveNum1 + positiveNum2;
        result.isPositive = num1.isPositive;
        return result;
    }

    // If num1 is smaller than num2, swap them and negate the result
    if (num1 < num2) {
        Bignum swappedNum1 = num2;
        Bignum swappedNum2 = num1;
        result = swappedNum1 - swappedNum2;
        result.isPositive = false;
        return result;
    }

    // Both numbers have the same sign, perform subtraction
    result.isPositive = num1.isPositive;
    result.tab.clear();
    uint64_t borrow = 0;

    // Iterate through digits of num1
    for (size_t i = 0; i < num1.tab.size(); i++) {
        uint64_t subtrahend = (i < num2.tab.size()) ? num2.tab[i] : 0;
        uint64_t tmp;

        // Perform subtraction with borrowing
        if (num1.tab[i] < subtrahend + borrow) {
            tmp = (BASE + num1.tab[i]) - (subtrahend + borrow);
            borrow = 1;
        }
        else {
            tmp = num1.tab[i] - (subtrahend + borrow);
            borrow = 0;
        }

        result.tab.push_back(tmp);
    }

    // Remove leading zeros in the result
    size_t i = result.tab.size() - 1;
    while (i != 0 && result.tab[i] == 0) {
        result.tab.pop_back();
        i--;
    }

    return result;
}

// Left-shift operator (<<) for Bignum
// Left-shift a Bignum by a specified number of bits.
Bignum operator<<(Bignum const& inputNum, unsigned long const& shiftAmount) {

    Bignum result = inputNum;
    unsigned long shift32Bits = shiftAmount / 32;
    unsigned long shift1Bit = shiftAmount % 32;

    // Shift the vector by 32 bits by inserting zeros at the beginning
    std::vector<uint32_t> shiftedVector;
    shiftedVector.reserve(result.tab.size() + shift32Bits);

    // Insert zeros at the beginning of the vector
    shiftedVector.insert(shiftedVector.end(), shift32Bits, 0x00);
    shiftedVector.insert(shiftedVector.end(), result.tab.begin(), result.tab.end());
    result.tab = shiftedVector;

    for (size_t bitIndex = 0; bitIndex < shift1Bit; bitIndex++) {
        uint32_t carry = 0;
        for (size_t i = 0; i < result.tab.size(); i++) {
            uint32_t newCarry = (result.tab[i] & 0x80000000) >> 31;
            result.tab[i] = (result.tab[i] << 1) | carry;
            carry = newCarry;
        }

        if (carry != 0) {
            result.tab.push_back(carry);
        }
    }

    return result;
}

// Right-shift operator (>>) for Bignum
Bignum operator>>(Bignum const& num, unsigned long const& shiftAmount) {
    Bignum result = num;
    unsigned long shift32 = shiftAmount / 32;
    unsigned long shift1 = shiftAmount % 32;

    // Erase elements at the beginning based on shift32
    if (shift32 > 0 && shift32 < result.tab.size()) {
        result.tab.erase(result.tab.begin(), result.tab.begin() + shift32);
    }
    // Clear the vector and add a zero if shift32 is equal to or greater than the size
    else if (shift32 >= result.tab.size()) {
        result.tab.clear();
        result.tab.push_back(0);  // Add a zero to avoid an empty vector
    }

    // Perform shift by shift1 bits
    for (size_t j = 0; j < shift1; j++) {
        uint32_t carry = 0;
        for (int i = result.tab.size() - 1; i >= 0; i--) {
            uint32_t newCarry = result.tab[i] & 0x00000001;
            result.tab[i] = (result.tab[i] >> 1);
            result.tab[i] |= (carry << 31);
            carry = newCarry;
        }
    }

    // Clean up non-significant zeros
    auto it = result.tab.end();
    while (it != result.tab.begin() && *(--it) == 0);

    // Erase unnecessary elements from the vector
    result.tab.erase(++it, result.tab.end());

    return result;
}

// Left-shift-assign operator (<<=) for Bignum
Bignum& Bignum::operator<<=(unsigned long const& x) {
    *this = *this << x;
    return *this;
}

// Right-shift-assign operator (>>=) for Bignum
Bignum& Bignum::operator>>=(unsigned long const& x) {
    *this = *this >> x;
    return *this;
}

// Multiplication operator (*) for Bignum
// Multiply two Bignums using Karatsuba multiplication.
Bignum operator*(const Bignum& num1, const Bignum& num2) {

    Bignum result;
    result.tab.clear();
    result.setPositive(!(num1.isPositive ^ num2.isPositive));

    size_t totalSize = num1.tab.size() + num2.tab.size();
    result.tab.resize(totalSize, 0);

    for (size_t i = 0; i < num1.tab.size(); ++i) {
        uint64_t carry = 0;
        for (size_t j = 0; j < num2.tab.size(); ++j) {
            uint64_t x = num1.tab[i];
            uint64_t y = num2.tab[j];

            // Perform Karatsuba multiplication
            uint64_t z0 = x * y;
            uint64_t z2 = x / BASE * y / BASE;
            uint64_t z1 = (x % BASE + x / BASE) * (y % BASE + y / BASE) - z0 - z2;

            // Update the result
            uint64_t tmp = carry + z0 + (z1 % BASE) + static_cast<uint64_t>(result.tab[i + j]);
            result.tab[i + j] = tmp % BASE;
            carry = (tmp / BASE) + (z1 / BASE);
        }
        result.tab[i + num2.tab.size()] += carry;
    }

    // Reduce the size of the result vector if necessary
    while (result.tab.size() > 1 && result.tab.back() == 0) {
        result.tab.pop_back();
    }

    return result;
}

// Function to perform division of two Bignums
// Divide two Bignums using long division.
std::pair<Bignum, Bignum> division(const Bignum& num1, const Bignum& num2) {

    Bignum quotient;  // q
    Bignum remainder = num1;  // r
    Bignum divisor = num2;
    Bignum bitShift = 1;

    // Handle the case where the divisor is zero
    if (num2 == Bignum(0)) {
        throw std::invalid_argument("Division by zero error");
    }

    // Handle the case where the dividend is zero
    if (num1 == Bignum(0)) {
        return std::make_pair(Bignum(0), Bignum(0));
    }

    // Handle the case where the divisor is greater than the dividend
    if (num1 < num2) {
        return std::make_pair(Bignum(0), remainder);
    }

    // Find the most significant bit in the divisor
    while (divisor <= remainder) {
        divisor <<= 1;
        bitShift <<= 1;
    }

    // Adjust one bit to the right
    divisor >>= 1;
    bitShift >>= 1;

    while (bitShift > 0) {
        quotient <<= 1;

        if (remainder >= divisor) {
            remainder = remainder - divisor;
            quotient = quotient + 1;
        }

        divisor >>= 1;
        bitShift >>= 1;
    }

    return std::make_pair(quotient, remainder);
}

// Division operator for Bignum
// Divide two Bignums using the division function and return the quotient.
Bignum operator/(Bignum const& x, Bignum const& y) {
    return division(x, y).first;
}

// Modulus operator for Bignum
// Divide two Bignums using the division function and return the remainder.
Bignum operator%(Bignum const& x, Bignum const& y) {
    return division(x, y).second;
}

// Modular exponentiation function using the square-and-multiply algorithm
Bignum mod_pow(const Bignum& base, const Bignum& exponent, const Bignum& modulo) {

    Bignum result = 1;
    Bignum base_copy = base % modulo;  // Reduce the base modulo modulo.
    Bignum exp_copy = exponent;  // Copy of the exponent for iteration.

    while (exp_copy > 0) {
        if (exp_copy % 2 == 1) {
            // If the exponent is odd, multiply by the base.
            result = (result * base_copy) % modulo;
        }
        // Reduce the exponent by half and update the base.
        exp_copy = (exp_copy / 2);
        base_copy = (base_copy * base_copy) % modulo;
    }

    return result;
}

// Function to calculate the bit length of a Bignum
int bitLength(Bignum n) {
    // Calculate the number of bits required to represent a Bignum.
    if (n == 0) {
        return 1; // If the number is zero, it requires one bit for representation.
    }
    int res = 0;
    Bignum temp = n;
    while (temp > 0) {
        temp = temp >> 1;
        res += 1;
    }

    return res;
}

// Function to generate a random Bignum in the range [1, n)
Bignum generateRandomBignum(const Bignum& n) {
    Bignum randomBignum = 1;

    // Initialize the random number generator with a random seed
    std::random_device rd;
    std::mt19937 gen(rd());

    // Use std::uniform_int_distribution to generate random bits
    std::uniform_int_distribution<uint32_t> distrib(0, 1);

    // Generate a random odd number using a loop
    do {
        randomBignum = 1; // Reset the number to 1 for a new attempt

        for (Bignum i = 1; i < n; i += 1) {
            // Generate a random bit and add it to randomBignum
            uint32_t randomBit = distrib(gen);
            randomBignum[0] |= randomBit;

            // Shift randomBignum one bit to the left
            randomBignum <<= 1;
        }

        // Ensure the number is odd
        randomBignum[0] |= 1;

    } while (randomBignum % 3 == 0 || randomBignum % 5 == 0);

    return randomBignum;
}

// Function to generate a random Bignum in the range [2, n-2)
Bignum generateRandomBignumRange(const Bignum& n) {
    Bignum randomBignum = 1;

    // Initialize the random number generator with a random seed
    std::random_device rd;
    std::mt19937 gen(rd());

    // Use std::uniform_int_distribution to generate random bits
    std::uniform_int_distribution<uint32_t> distrib(0, 1);

    // Generate a random odd number with a loop
    do {
        randomBignum = 1; // Reset the number to 1 for a new attempt

        for (Bignum i = 2; i < n - 2; i += 1) {
            // Generate a random bit and add it to randomBignum
            uint32_t randomBit = distrib(gen);
            randomBignum[0] |= randomBit;

            // Shift randomBignum one bit to the left
            randomBignum <<= 1;
        }

        // Ensure the number is odd
        randomBignum[0] |= 1;

    } while (randomBignum % 3 == 0 || randomBignum % 5 == 0);

    return randomBignum;
}

// Function to perform the Miller-Rabin primality test
bool millerRabinTest(const Bignum& n, int k) {
    Bignum size_of_n = bitLength(n);

    // Check if n is less than or equal to 1, or even
    if (n <= 1 || n % 2 == 0) {
        return false;
    }

    // Handle small values of n directly
    if (n <= 3) {
        return true;
    }

    // For n > 3, perform the Miller-Rabin test
    if (size_of_n <= 128) {
        // Use smaller bases for basic primality testing
        for (const int base : {2, 3, 5, 7}) {
            if (mod_pow(base, n - 1, n) != 1) {
                return false;
            }
        }
        return true;
    }
    else {
        Bignum d = n - 1;
        int r = 0;
        while (d % 2 == 0) {
            d /= 2;
            r++;
        }

        // Perform the Miller-Rabin test with k iterations
        for (int i = 0; i < k; ++i) {
            // Choose a random integer a in the range [2, n-2]
            Bignum a = generateRandomBignumRange(size_of_n - 2);

            // Calculate a^d mod n
            Bignum x = mod_pow(a, d, n);

            // If x is 1 or n-1, move to the next iteration
            if (x == 1 || x == n - 1) {
                continue;
            }

            // Iterate the Miller-Rabin test
            for (int j = 0; j < r - 1; ++j) {
                x = (x * x) % n;
                if (x == n - 1) {
                    break;
                }
            }

            // If none of the conditions are satisfied, n is composite
            if (x != n - 1) {
                return false;
            }
        }

        // If all iterations pass, n is probably prime
        return true;
    }
}

// Function to generate a random prime number of the desired bit length
Bignum getRandomPrime(Bignum const& desiredBitLength) {
    Bignum two("2");
    Bignum three("3");

    // Generate a random odd number with the desired bit length
    Bignum candidatePrime = generateRandomBignumRange(desiredBitLength);

    // Ensure candidatePrime is odd
    if (candidatePrime % two == 0) {
        candidatePrime += 1;
    }

    // Apply the algorithm to find the next prime number
    while (true) {
        if (millerRabinTest(candidatePrime, 15)) {
            return candidatePrime; // Prime number found
        }

        candidatePrime += two; // Move to the next odd number

        // Ensure candidatePrime is not divisible by 3
        if (candidatePrime % three == 0) {
            candidatePrime += two; // Move to the next odd number
        }
    }
}


ostream& operator<<(ostream& flux, Bignum const& x) {
    if (x == Bignum(0)) flux << 0;
    else {
        if (!x.isPositive) flux << "-";
        auto p = division(x, 10);
        p.first.isPositive = true;
        if (p.first != Bignum(0)) flux << p.first;
        flux << p.second[0];
    }
    return flux;
}

// Extended Euclidean Algorithm (EEA) to find GCD and coefficients x, y such that ax + by = GCD(a, b)
std::tuple<Bignum, Bignum, Bignum> EEA(Bignum a, Bignum b) {
    Bignum x0(1), x1(0), y0(0), y1(1);

    while (b != 0) {
        Bignum q = a / b;
        Bignum r = a % b;

        Bignum x2 = x0 - q * x1;
        Bignum y2 = y0 - q * y1;

        x0 = x1;
        y0 = y1;
        x1 = x2;
        y1 = y2;

        a = b;
        b = r;
    }

    // Ensure that x0 is positive
    if (x0 < 0) {
        x0 += b;
    }

    return std::make_tuple(a, x0, y0);
}

// Modular Inverse using Extended Euclidean Algorithm
Bignum modularInverse(const Bignum& a, const Bignum& n) {
    Bignum x, y, gcd;
    std::tie(gcd, x, y) = EEA(a, n);

    // Ensure that x is positive
    if (x < 0) {
        x += n;
    }

    if (gcd == 1) {
        return x; // Return the modular inverse
    }
    else {
        // a does not have an inverse modulo n
        return Bignum(0);
    }
}

// Function to generate public and private key pairs for RSA encryption
std::pair<std::pair<Bignum, Bignum>, std::pair<Bignum, Bignum>> generateKeys(int keySize) {
    std::cout << "\nGenerating encryption keys, please wait..." << std::endl;

    // Choose two prime numbers
    Bignum prime1 = getRandomPrime(keySize);
    Bignum prime2 = getRandomPrime(keySize);

    std::cout << "Prime numbers selected." << std::endl;

    // Calculate n and phi
    Bignum n = prime1 * prime2;
    Bignum phi = (prime1 - 1) * (prime2 - 1);

    // Choose the public exponent e
    Bignum publicExponent("65537"); // A common choice for e
    Bignum privateExponent = modularInverse(publicExponent, phi);

    // Build the pairs of public and private keys
    std::pair<Bignum, Bignum> publicKey(publicExponent, n);
    std::pair<Bignum, Bignum> privateKey(privateExponent, n);

    std::pair<std::pair<Bignum, Bignum>, std::pair<Bignum, Bignum>> keyPair(publicKey, privateKey);

    std::cout << "Encryption keys generated successfully." << std::endl;

    return keyPair;
}

// Function to encode a string into a vector of Bignums
std::vector<Bignum> encodeData(const std::string& input, int blockSize) {
    int count = 0;
    std::vector<Bignum> result;
    Bignum segment(0);

    // Iterate through each character in the input string
    for (size_t i = 0; i < input.length(); i++) {
        // Shift the segment by 8 bits and add the ASCII value of the character
        segment <<= 8;
        segment += input[i];
        count++;

        // If the count reaches the specified block size, add the segment to the result
        if (count == (blockSize / 8)) {
            result.push_back(segment);
            segment = 0;
            count = 0;
        }
    }

    // If there's any remaining data in the segment, add it to the result
    if (segment != 0) {
        result.push_back(segment);
    }

    return result;
}

// Function to decode a vector of Bignums into a string
std::string decodeData(std::vector<Bignum> encoded) {
    std::string result;

    // Iterate until the last Bignum is greater than 0
    while (encoded.front() > 0) {
        // Remove trailing zeros from the last Bignum
        if (encoded.back() == 0) {
            encoded.pop_back();
        }

        // Extract the last byte (8 bits) from the last Bignum
        Bignum lastElement = encoded.back() % Bignum(256);
        // Convert the ASCII value to a character and prepend to the result string
        result = static_cast<char>(lastElement.tab.front()) + result;

        // Shift the last Bignum to the right by 8 bits
        encoded.back() = encoded.back() >> 8;
    }

    return result;
}

// Function to encrypt a vector of Bignums using RSA
std::vector<Bignum> encrypt(std::vector<Bignum> plaintext, Bignum modulus, Bignum privateKey) {
    std::vector<Bignum> encryptedData;
    Bignum ciphertext(0);

    // Iterate through each block of plaintext
    for (const Bignum& block : plaintext) {
        // Use modular exponentiation to compute the ciphertext
        ciphertext = mod_pow(block, privateKey, modulus);
        encryptedData.push_back(ciphertext);
    }

    return encryptedData;
}

// Function to decrypt a vector of Bignums using RSA
std::vector<Bignum> decrypt(std::vector<Bignum> ciphertext, Bignum modulus, Bignum publicKey) {
    std::vector<Bignum> decryptedData;
    Bignum plaintext(0);

    // Iterate through each block of ciphertext
    for (const Bignum& block : ciphertext) {
        // Use modular exponentiation to compute the plaintext
        plaintext = mod_pow(block, publicKey, modulus);
        decryptedData.push_back(plaintext);
    }

    return decryptedData;
}


int main() {
    int choice;

    std::cout << "Choose 1) Encrypt and Decrypt, 2) Encrypt only ? ";
    std::cin >> choice;

    if (choice == 1) {
        // Option 1: Encrypt and Decrypt
        int keysize;

        std::cout << "Enter the key size in bits: ";
        std::cin >> keysize;

        std::string message;

        // Get message to encrypt from user
        std::cout << "Enter the message you want to encrypt: ";
        std::cin.ignore();
        std::getline(std::cin, message);

        std::string privateKeyChoice;

        // Option to display private key
        std::cout << "Do you want to display the private key ? (yes/no): ";
        std::cin >> privateKeyChoice;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Generate key pair
        std::pair<std::pair<Bignum, Bignum>, std::pair<Bignum, Bignum>> keyPair = generateKeys(keysize);
        std::pair<Bignum, Bignum> publicKey = keyPair.first;
        std::pair<Bignum, Bignum> privateKey = keyPair.second;

        auto encryption_start_time = std::chrono::high_resolution_clock::now();

        // Encrypt the message
        std::vector<Bignum> encrypted = encrypt(encodeData(message, keysize), publicKey.second, publicKey.first);

        // Display keys and encrypted message
        std::cout << "\nPublic Key: " << publicKey.first << ", " << publicKey.second << "\n";
        if (privateKeyChoice == "yes") {
            std::cout << "Private Key: " << privateKey.first << ", " << privateKey.second << "\n";
        }
        std::cout << "\nEncrypted Message: " << encrypted[0] << "\n";

        // Decrypt the message and display
        std::string decrypted = decodeData(decrypt(encrypted, privateKey.second, privateKey.first));
        std::cout << "Decrypted Message: " << decrypted << "\n";

        // Calculate and display time taken for encryption and decryption
        auto encryption_end_time = std::chrono::high_resolution_clock::now();
        auto encryption_duration = std::chrono::duration_cast<std::chrono::milliseconds>(encryption_end_time - encryption_start_time);
        auto keyduration = std::chrono::duration_cast<std::chrono::milliseconds>(encryption_start_time - start_time);
        std::cout << "\nTime taken to generate Keys: " << keyduration.count() << " milliseconds\n";
        std::cout << "Time taken for encryption and decryption: " << encryption_duration.count() << " milliseconds\n";

    }
    else if (choice == 2) {
        // Option 2: Encrypt only
        int keysize;

        std::cout << "Enter the key size in bits: ";
        std::cin >> keysize;

        std::string message;

        // Get message to encrypt from user
        std::cout << "Enter the message you want to encrypt: ";
        std::cin.ignore();
        std::getline(std::cin, message);

        std::string privateKeyChoice;

        // Option to display private key
        std::cout << "Do you want to display the private key ? (yes/no): ";
        std::cin >> privateKeyChoice;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Generate key pair
        std::pair<std::pair<Bignum, Bignum>, std::pair<Bignum, Bignum>> keyPair = generateKeys(keysize);
        std::pair<Bignum, Bignum> publicKey = keyPair.first;

        auto encryption_start_time = std::chrono::high_resolution_clock::now();

        // Encrypt the message
        std::vector<Bignum> encrypted = encrypt(encodeData(message, keysize), publicKey.second, publicKey.first);

        // Display keys and encrypted message
        std::cout << "\nPublic Key: " << publicKey.first << ", " << publicKey.second << "\n";
        if (privateKeyChoice == "yes") {
            std::cout << "Private Key: " << keyPair.second.first << ", " << keyPair.second.second << "\n";
        }
        std::cout << "Encrypted Message: " << encrypted[0] << "\n";

        // Calculate and display time taken for encryption
        auto encryption_end_time = std::chrono::high_resolution_clock::now();
        auto encryption_duration = std::chrono::duration_cast<std::chrono::milliseconds>(encryption_end_time - encryption_start_time);
        auto keyduration = std::chrono::duration_cast<std::chrono::milliseconds>(encryption_start_time - start_time);
        std::cout << "\nTime taken to generate Keys: " << keyduration.count() << " milliseconds\n";
        std::cout << "Time taken for encryption: " << encryption_duration.count() << " milliseconds\n";
    }
    else {
        std::cout << "Invalid choice. Please choose 1 or 2.\n";
    }

    return 0;
}
