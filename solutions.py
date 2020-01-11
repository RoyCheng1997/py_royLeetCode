# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 12:57:04 2020

Leet Code Practice

1,31,54,76,78,
131,
202,204,263,264,278,289,
300,
-
-
643,


@author: RoyCheng
"""

# %% 
# No. 0001 Two Sum     
# Difficulty: Easy
# Tag: -Hash
# Comments: 
# Complexity: O(n)
class Solution0001(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashmap = {}
        for idx, num in enumerate(nums):
            rem  = target - num
            if rem in hashmap:
                return [idx, hashmap[rem]]
            hashmap[num] = idx
#%%
# No. 0031 Next Permutation      
# Difficulty: Medium
# Tag: -Pattern
# Comments: special algorithm
# Complexity: O(n)
class Solution0031:
    def nextPermutation(self, nums):
        num = 0
        index = -1
        # find first decreasing number
        for i in range(len(nums) - 1, 0, -1):
            if nums[i-1] < nums[i]:
                num = nums[i-1]
                index = i-1
                break
        
        if index == -1: # already in reversed order
            nums.reverse()
        else:
            prev = float('inf')
            p_index = -1 
            # find first number large than the nums[index]
            for i in range(index+1, len(nums)):
                if nums[i] > num and nums[i] <= prev:
                    prev = nums[i]
                    p_index = i
            # swap and sort the rest
            nums[index], nums[p_index] = nums[p_index], nums[index]
            nums[index+1:] = reversed(nums[index+1:])

#%%
# No. 0054 Spiral Matrix      
# Difficulty: Medium
# Tag: -Pattern
# Comments: simulation method
# Complexity: O(n), n total # of elements
class Solution0054(object):
    def spiralOrder(self, matrix):
        if not matrix: return []
        R, C = len(matrix), len(matrix[0]) # number of rows and columns
        seen = [[False] * C for _ in matrix] # visitd ?
        result = []
        direction_row = [0, 1, 0, -1]
        direction_col = [1, 0, -1, 0]
        r = c = direction = 0 # current location
        for _ in range(R * C):
            result.append(matrix[r][c])
            seen[r][c] = True
            tmp_row, tmp_col = r + direction_row[direction], c + direction_col[direction] # candidate next position
            if 0 <= tmp_row < R and 0 <= tmp_col < C and not seen[tmp_row][tmp_col]: # in the bounds of the matrix and unseen
                r, c = tmp_row, tmp_col
            else: # performing a clockwise turn
                direction = (direction + 1) % 4
                r, c = r + direction_row[direction], c + direction_col[direction]
        return result
#%%
# No.0076 Minimum Window Substring     
# Difficulty: Hard
# Tag: - double pointers
# Comments: 
# Complexity:O(len(s)+len(t)) 
class Solution0076(object):
    def minWindow(self, s, t): # double pointers method, Sliding Window
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if not t or not s:
            return ""
        dict_t = self.__count_unique__(t)
        totalUniqueChar = len(dict_t)# Number of unique characters in t,
        l, r = 0, 0 # left and right pointer
        includeUniqueChar = 0
        window_counts = {} # same format as dict_t
        # result tuple of the form (window length, left, right)
        result = float("inf"), None, None
        while r < len(s):
            character = s[r] # inlude one new on right
            window_counts[character] = window_counts.get(character,0) + 1
            # If the frequency of the current character added equals to the desired count in t then increment the formed count by 1.
            if character in dict_t and window_counts[character] == dict_t[character]:
                includeUniqueChar += 1
            # Try and contract the window till the point where it ceases to be 'desirable'.
            while l <= r and includeUniqueChar == totalUniqueChar:
                character = s[l]
                if r - l + 1 < result[0]: # Save the smallest window until now.
                    result = (r - l + 1, l, r)
                # The character at the position pointed by the `left` pointer is no longer a part of the window.
                window_counts[character] -= 1
                if character in dict_t and window_counts[character] < dict_t[character]:
                    includeUniqueChar -= 1
                # Move the left pointer ahead, this would help to look for a new window.
                l += 1    
            # Keep expanding the window once we are done contracting.
            r += 1    
        return "" if result[0] == float("inf") else s[result[1] : result[2] + 1]       

    def __count_unique__(self, string):
        '''count unique characters in string'''
        dic = {}
        for i in range(len(string)):
            if string[i] not in dic.keys():
                dic[string[i]] = string.count(string[i])
            else:
                pass
        return dic
    
    def SharePurchase(s): # special t for 'ABC'
        dic = dict(zip('ABC'),[0,0,0])
        left = 0
        count = 0
        for right in range(len(s)):
            if s[right] in 'ABC':
                dic[s[right]] += 1
                while all(dic.values()):
                    count += len(s) - right
                    if s[left] in 'ABC':
                        dic[s[left]] -= 1
                    left += 1
        return count
#%%
# No. 0078 Subsets      
# Difficulty: Medium
# Tag: -backtracking, -recursion
# Comments: recursion solution
# Complexity: O(n*2^n)
class Solution0078:
    def subsets(self, nums):
        return self.__recursion__([[]],nums)
    def __recursion__(self,res, nums):
        if not nums:
            return res
        newres = []
        for sett in res:
            newres.append(sett + [nums[0]])
        return self.recursion(newres + res, nums[1:])    
    def subsets1(self, nums): # same algo, simple notation
        #n = len(nums)
        output = [[]]
        for num in nums:
            output += [curr + [num] for curr in output]
        return output    
#%%
# No. 0131 Palindrome Partitioning       
# Difficulty: Medium
# Tag: -DP, -recursion
# Comments:
# Complexity: NAN
class Solution0131:
    def partition_recur(self, s: str) -> List[List[str]]:
        self.partitions = []
        self.backtrack(0, (), s)
        return self.partitions
    
    def backtrack(self, pos: int, partition: tuple, s:str):
        if pos == len(s):
            self.partitions.append(list(partition))
        else:
            # Traverse from `pos` to the end of the string
            for i in range(pos, len(s)):
                # If the substring is a palindrome then add it 
                # to the current parition and recusively backtrack
                # from `i+1`
                if self.is_palindromic(s[pos:i+1]):
                    self.backtrack(i+1, partition + (s[pos:i+1],), s)
                    
    def is_palindromic(self, s: str) -> bool: 
        ''' whether is palindromic, symmetric'''
        return s[:] == s[::-1]    
    
    def partition_DP(self, s: str) -> List[List[str]]:
        dp = [[[]]]
        psi = [0]   # palindrome start indices
        for i, c in enumerate(s):
            psi = [k for k in psi if s[k] == s[i]]
            dp.append([pp + [s[k:i+1]] for k in psi for pp in dp[k]])
            psi = [k-1 for k in psi if k > 0] + [i, i+1]
        return dp[-1]
    
#%%
# No. 0202 Happy Number      
# Difficulty: Easy
# Tag: -recursion
# Comments: use space in exchange for time
# Complexity: NAN
class Solution0202:
    def isHappy(self, n: int) -> bool: # brute force
        boolvalue = False
        numberList = [n]
        number = n
        while (boolvalue == False):
            # calculate
            sumS = 0
            for i in range(0,len(str(number))):
                sumS += int(str(number)[i]) * int(str(number)[i])
            # exit condi.
            if (sumS == 1):
                boolvalue = True
                return boolvalue
            elif (sumS in numberList):
                boolvalue = False
                return boolvalue
            else:
                numberList.append(sumS)
                number = sumS
    
    def isHappy_recursion(self, n: int) -> bool:
        makeList = [] # extra list which keeps track of numbers that have appeared in the calculation
        return self.__testIsHappy__(makeList, n)
    
    def __testIsHappy__ (self, theList: list, n: int) -> bool:
        if n == 1:
            return True
        for i in theList: # if in the extra list
            if i == n:
                return False
        theList.append(n) # store unhappy one
        num = str(n)
        total = 0
        for x in range(len(num)):
            total += int(num[x]) * int(num[x])
        return self.testIsHappy(theList, total)   
    
#%%
# No. 0204 Count Prime    
# Difficulty: Easy
# Tag: -typical
# Comments: Sieve method
# Complexity: O(nlog(log(n)))
class Solution0264:
	def countPrimes(self, n: int) -> int:# sieve method
		# https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
		if n <= 2:
			return 0
		else:
			markers = [1] * n # to mark an element from 0 to n-1
			markers[0] = 0
			markers[1] = 0
			# iterate
			m = 2
			while m*m < n: # less than n
				if markers[m] == 1: # found a prime, then its multiples are not prime
					# mark from m * m, as m * (m-1), m * (m-2), and so on have been marked in previous iteration
					markers[m*m:n:m] = [0] * (1 + (n-1-m*m)//m)
				m += 1
			return sum(markers)

# prime number
def isPrime(num):
    '''to judge whether the number is prime (best solution)'''
    if (num <= 3): return num>1
    if (num % 6 !=1) and (num % 6 !=5): #6k-1 or 6k+5
        return False
    sqrtValue = int(sqrt(num))
    for i in range(5,sqrtValue+1,6):
        if (num % i == 0) or (num % (i+2) == 0):
            return False
    return True

#%%
# No. 0263 Ugly Number      
# Difficulty: Easy
# Tag: 
# Comments:
# Complexity: O(log(n))
class Solution0263:
    def isUgly(self, num: int) -> bool: # loop
        if num == 1:
            return True
        if num <= 0:
            return False
        while(num > 1):
            if num % 2 != 0 and num % 3 != 0 and num % 5 != 0:
                return False
            if num % 2 == 0:
                num = num / 2
            elif num % 3 == 0:
                num = num / 3
            elif num % 5 == 0:
                num = num / 5
        return True    
#%%
# No. 0264 Ugly Number 2     
# Difficulty: Medium
# Tag: -DP
# Comments: use space in exchange for time
# Complexity: O(n)
class Solution0264:
    def nthUglyNumbern(self, n: int) -> int:
        c2,c3,c5 = 0,0,0 # count of 2,3,5
        n2,n3,n5 = 2,3,5
        dpList = [1] # result list
        while len(dpList)<n:
            x = min(n2,min(n3,n5)) # min(n2,n3,n5),possible solution for next iteration
            if x != dpList[-1]: # if not in it
                dpList.append(x)
            else:
                pass
            # iteration
            if x == n2: 
                c2 += 1
                n2 = 2 * dpList[c2]
            elif x == n3:
                c3 += 1
                n3 = 3 * dpList[c3]
            else:
                c5 += 1
                n5 = 5 * dpList[c5]
        return dpList[n-1]

    
#%%
# No. 0278 First Bad Version      
# Difficulty: Easy
# Tag: -typical
# Comments: 
# Complexity: O(log(n))
class Solution0278:
    def firstBadVersion(self, n): # binary search 
        """
        :type n: int
        :rtype: int
        """
        left,right = 1,n
        while (left < right):
            mid = int((left+right)/2)
            if (isBadVersion(mid)): # built in method
                right = mid
            else:
                left = mid + 1
        return left        

#%%
# No. 0289 Game of Life      
# Difficulty: Medium
# Tag: -typical
# Comments: one stage game of life
# Complexity: O(m*n)
import copy
class Solution0289:
    def gameOfLife(self, board): # space solution
        R,C = len(board),len(board[0])
        board_copy = copy.deepcopy(board)
        nb_coordinate = [(1,0),(1,-1),(1,1),(0,-1),(0,1),(-1,0),(-1,-1),(-1,1)]
        for row in range(R):
            for col in range(C):
                live_num = self.__calNeiborLive__((row,col),nb_coordinate,board_copy,R,C)
                # Rule 1 or Rule 3        
                if board_copy[row][col] == 1 and (live_num < 2 or live_num > 3):
                    board[row][col] = 0
                # Rule 4
                if board_copy[row][col] == 0 and live_num == 3:
                    board[row][col] = 1    
        return board
    
    def __calNeiborLive__(self,cor,nb_cor,board_copy,R,C):
        live = 0
        for item in nb_cor:
                nb_row = cor[0] + item[0]
                nb_col = cor[1] + item[1]
                if (nb_row < R and nb_row >= 0) and (nb_col < C and nb_col >= 0) and (board_copy[nb_row][nb_col] == 1):
                    live += 1        
                else:
                    pass
        return live
    
    def grid_game(grid, k, rules): # loop solution
        m = len(grid)
        n = len(grid[0])
        rule_list = [i for i, x in enumerate(rules) if x == "alive"]
        neighbors = [(1, 1), (0, 1), (1, 0), (-1, 0), (0, -1), (-1, -1), (1, -1), (-1, 1)]
        for _ in range(k, 0, -1):
            for row in range(m):
                for col in range(n):
                    live_neighbor = 0
                    for neighbor in neighbors:
                        temp_row = row + neighbor[0]
                        temp_col = col + neighbor[1]
    
                        if m > temp_row >= 0 and n > temp_col >= 0 and abs(grid[temp_row][temp_col]) == 1:
                            live_neighbor += 1
    
                    if grid[row][col] == 1 and live_neighbor not in rule_list:
                        grid[row][col] = -1
                    if grid[row][col] == 0 and live_neighbor in rule_list:
                        grid[row][col] = 2
            for row in range(m):
                for col in range(n):
                    if grid[row][col] > 0:
                        grid[row][col] = 1
                    else:
                        grid[row][col] = 0
                        
#%% 
# No. 0300 Longest Increasing Subsequence    
# Difficulty: Medium
# Tag: -DP
# Comments: 
# Complexity: O(n^2)        
class Solution0300:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        dp = [0] * len(nums) #  length of the longest increasing subsequence possible considering the array elements upto the i th index only
        dp[0] = 1
        maxans = 1 # result
        for i in range(1,len(nums)):
        # by necessarily including the ith element. 
        # In order to find out dp[i] we need to try to append the current element(nums[i]) in every possible increasing 
        # subsequences upto the (i-1) th elemets
        # such that the new sequence formed by adding the current element is also an increasing subsequence
            maxval = 0 # dp[i]=max(dp[j])+1, for all 0≤j<i
            for j in range(0,i):
                if nums[i] > nums[j]: # else maxval=0
                    maxval = max(maxval,dp[j])
            dp[i] = maxval +1
            maxans =  max(maxans,dp[i]) # at the end
        return maxans
                        
#%%
# No. 0643 Maximum Average Subarray I     
# Difficulty: Easy
# Tag: -typical
# Comments: 
# Complexity: O(n)
class Solution0643(object):
    def findMaxAverage(self, nums, k): # cumulative sum
        """
        :type nums: List[int]
        :type k: int
        :rtype: float
        """
        sumList = [nums[0]]
        for i in range(1,len(nums)):
            sumList.append(nums[i]+sumList[-1])
        result = sumList[k - 1] * 1.0 / k;
        for i in range(k,len(nums)):
		    result = max(result, (sumList[i] - sumList[i - k]) * 1.0 / k);
        return result

#%%
# No.      
# Difficulty:
# Tag:
# Comments: 
# Complexity:
#%%
# No.      
# Difficulty:
# Tag:
# Comments: 
# Complexity:
#%%
# No.      
# Difficulty:
# Tag:
# Comments: 
# Complexity:
#%%
# No.      
# Difficulty:
# Tag:
# Comments: 
# Complexity:
#%%
# No.      
# Difficulty:
# Tag:
# Comments: 
# Complexity:
#%%
# No.      
# Difficulty:
# Tag:
# Comments: 
# Complexity:
#%%
# No.      
# Difficulty:
# Tag:
# Comments: 
# Complexity:
#%%
# No.      
# Difficulty:
# Tag:
# Comments: 
# Complexity:
#%%
# No.      
# Difficulty:
# Tag:
# Comments: 
# Complexity:
#%%
# No.      
# Difficulty:
# Tag:
# Comments: 
# Complexity:
#%%
# No.      
# Difficulty:
# Tag:
# Comments: 
# Complexity: