# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 12:57:04 2020

Leet Code Practice

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
# Tag:
# Comments: 
# Complexity:
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
# No. 0278 First Bad Version      
# Difficulty: Easy
# Tag: -typical
# Comments: 
# Complexity: O(log(n))
class Solution:
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
class Solution(object):
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
#%%
# No. 0643 Maximum Average Subarray I     
# Difficulty: Easy
# Tag: -typical
# Comments: 
# Complexity: O(n)
class Solution(object):
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