import React, { useState, useEffect } from 'react';
import { Star, ShoppingCart, MapPin, Search, Menu, ChevronDown, Heart } from 'lucide-react';

const API_URL = 'http://localhost:5000/api';

// Fallback categories if backend not ready
const FALLBACK_CATEGORIES = [
  'Electronics',
  'Beauty_and_Personal_Care',
  'Books',
  'Sports_and_Outdoors',
  'Baby_Products'
];

// Category display names (prettier formatting)
const CATEGORY_DISPLAY_NAMES = {
  'Electronics': 'Electronics',
  'Beauty_and_Personal_Care': 'Beauty & Personal Care',
  'Books': 'Books',
  'Sports_and_Outdoors': 'Sports & Outdoors',
  'Baby_Products': 'Baby Products'
};

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [token, setToken] = useState(null);
  const [username, setUsername] = useState('');
  const [showAuth, setShowAuth] = useState(false);
  const [isRegister, setIsRegister] = useState(false);
  const [authForm, setAuthForm] = useState({ username: '', password: '', email: '' });
  const [selectedCategory, setSelectedCategory] = useState('Electronics');
  const [categories, setCategories] = useState(FALLBACK_CATEGORIES); // Use fallback initially
  const [recommendations, setRecommendations] = useState([]);
  const [strategy, setStrategy] = useState('');
  const [userScenario, setUserScenario] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [showCategoryMenu, setShowCategoryMenu] = useState(false);
  const [showColdItems, setShowColdItems] = useState(false);
  const [groupedColdItems, setGroupedColdItems] = useState({});

  // Helper function to get display name
  const getCategoryDisplayName = (category) => {
    return CATEGORY_DISPLAY_NAMES[category] || category.replace(/_/g, ' & ');
  };

  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    const savedUsername = localStorage.getItem('username');
    if (savedToken && savedUsername) {
      setToken(savedToken);
      setUsername(savedUsername);
      setIsLoggedIn(true);
    }
    fetchCategories();
  }, []);

  useEffect(() => {
    if (selectedCategory) {
      if (showColdItems) {
        fetchColdItems();
      } else {
        fetchRecommendations();
      }
    }
  }, [isLoggedIn, selectedCategory, showColdItems]);

  const fetchCategories = async () => {
    try {
      const response = await fetch(`${API_URL}/categories`);
      const data = await response.json();
      
      if (data.categories && data.categories.length > 0) {
        setCategories(data.categories);
        // Set first category if current selection not in list
        if (!data.categories.includes(selectedCategory)) {
          setSelectedCategory(data.categories[0]);
        }
      } else {
        // Use fallback if backend returns empty
        setCategories(FALLBACK_CATEGORIES);
      }
    } catch (err) {
      console.error('Failed to fetch categories:', err);
      // Use fallback on error
      setCategories(FALLBACK_CATEGORIES);
    }
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    
    const endpoint = isRegister ? '/register' : '/login';
    const payload = isRegister 
      ? { username: authForm.username, password: authForm.password, email: authForm.email }
      : { username: authForm.username, password: authForm.password };

    try {
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      
      if (response.ok) {
        if (isRegister) {
          setError('Registration successful! Please sign in.');
          setIsRegister(false);
          setAuthForm({ username: '', password: '', email: '' });
        } else {
          setToken(data.access_token);
          setUsername(data.username);
          setIsLoggedIn(true);
          localStorage.setItem('token', data.access_token);
          localStorage.setItem('username', data.username);
          setShowAuth(false);
          setAuthForm({ username: '', password: '', email: '' });
        }
      } else {
        setError(data.error || `${isRegister ? 'Registration' : 'Login'} failed`);
      }
    } catch (err) {
      setError('Network error. Please check your connection.');
    }
    setLoading(false);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setToken(null);
    setUsername('');
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    setRecommendations([]);
  };

  const fetchRecommendations = async () => {
    setLoading(true);
    setError('');
    try {
      const headers = {};
      if (token) headers['Authorization'] = `Bearer ${token}`;
  
      const response = await fetch(`${API_URL}/recommendations/${selectedCategory}?n=20`, { headers });
      const data = await response.json();
  
      if (data.recommendations && Array.isArray(data.recommendations)) {
        const boostedRecs = boostRatedItems(data.recommendations, selectedCategory);
        setRecommendations(boostedRecs);
        setStrategy(data.strategy || '');
        setUserScenario(data.user_scenario || null);
      } else {
        setError(data.error || 'No recommendations available');
      }
    } catch (err) {
      setError('Network error');
    }
    setLoading(false);
  };

  const handleRate = async (parent_asin, rating) => {
    try {
      const response = await fetch(`${API_URL}/rate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ parent_asin, rating, category: selectedCategory })
      });
      
      if (response.ok) {
        const userRatings = JSON.parse(localStorage.getItem('userRatings') || '{}');
        userRatings[parent_asin] = { rating, timestamp: Date.now(), category: selectedCategory };
        localStorage.setItem('userRatings', JSON.stringify(userRatings));
        
        // Refresh data based on current view
        if (showColdItems) {
          await fetchColdItems();
        } else {
          await fetchRecommendations();
        }
      }
    } catch (err) {
      console.error('Failed to rate product:', err);
    }
  };

  const fetchColdItems = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${API_URL}/cold-items/${selectedCategory}`);
      const data = await response.json();
      
      if (data.grouped_items) {
        setGroupedColdItems(data.grouped_items);
      } else {
        setError(data.error || 'No cold items available');
        setGroupedColdItems({});
      }
    } catch (err) {
      setError('Network error');
      setGroupedColdItems({});
    }
    setLoading(false);
  };

  const boostRatedItems = (recs, selectedCategory) => {
    const userRatings = JSON.parse(localStorage.getItem('userRatings') || '{}');
    if (Object.keys(userRatings).length === 0) return recs;
    
    const boostedRecs = recs.map(item => {
      const userRating = userRatings[item.parent_asin];
      if (userRating && userRating.category === selectedCategory) {
        const boostFactor = userRating.rating / 5;
        const newScore = Math.round(item.score * (1 + boostFactor) * 100) / 100;
        return { ...item, score: newScore, user_rated: true, user_rating: userRating.rating };
      }
      return item;
    });
    
    return boostedRecs.sort((a, b) => b.score - a.score);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Amazon Header */}
      <header className="sticky top-0 z-50">
        {/* Main Navigation */}
        <div className="bg-[#131921]">
          <div className="max-w-[1500px] mx-auto px-4">
            <div className="flex items-center justify-between h-16">
              {/* Logo */}
              <div className="flex items-center space-x-6">
                <div className="cursor-pointer hover:border border-white p-2 rounded transition">
                  <div className="text-3xl font-bold">
                    <span className="text-white">amazon</span>
                    <span className="text-[#FF9900]">.ml</span>
                  </div>
                </div>
                
                <div className="hidden lg:flex items-center space-x-1 cursor-pointer hover:border border-white p-2 rounded">
                  <MapPin className="w-5 h-5 text-white" />
                  <div className="flex flex-col text-xs text-white">
                    <span className="text-gray-400">Deliver to</span>
                    <span className="font-bold">Boston 02115</span>
                  </div>
                </div>
              </div>

              {/* Search Bar */}
              <div className="flex-1 max-w-3xl mx-4">
                <div className="flex rounded-md overflow-hidden shadow-lg">
                  <button 
                    onClick={() => setShowCategoryMenu(!showCategoryMenu)}
                    className="bg-gray-200 text-gray-700 px-4 py-2.5 hover:bg-gray-300 flex items-center space-x-1 text-sm relative"
                  >
                    <span className="max-w-[120px] truncate">
                      {getCategoryDisplayName(selectedCategory)}
                    </span>
                    <ChevronDown className="w-4 h-4 flex-shrink-0" />
                    {showCategoryMenu && (
                      <div className="absolute top-full left-0 bg-white mt-1 rounded shadow-xl py-2 w-72 max-h-96 overflow-y-auto z-50">
                        <div className="px-4 py-2 text-xs font-bold text-gray-500 border-b">SHOP BY DEPARTMENT</div>
                        {categories.map(cat => (
                          <div
                            key={cat}
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedCategory(cat);
                              setShowCategoryMenu(false);
                            }}
                            className={`px-4 py-2.5 hover:bg-gray-100 cursor-pointer text-sm border-l-2 ${
                              selectedCategory === cat ? 'border-orange-500 bg-gray-50 font-medium' : 'border-transparent'
                            }`}
                          >
                            {getCategoryDisplayName(cat)}
                          </div>
                        ))}
                      </div>
                    )}
                  </button>
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search Amazon"
                    className="flex-1 px-4 py-2.5 text-gray-900 outline-none text-sm"
                  />
                  <button className="bg-[#FEBD69] px-6 py-2.5 hover:bg-[#F3A847] transition">
                    <Search className="w-5 h-5 text-gray-900" />
                  </button>
                </div>
              </div>

              {/* Right Nav */}
              <div className="flex items-center space-x-2">
                {isLoggedIn ? (
                  <>
                    <div className="hidden md:flex flex-col text-xs cursor-pointer hover:border border-white p-2 rounded">
                      <span className="text-gray-300">Hello, {username}</span>
                      <span className="font-bold text-white">Account & Lists</span>
                    </div>
                    <div className="hidden md:flex flex-col text-xs cursor-pointer hover:border border-white p-2 rounded">
                      <span className="text-gray-300">Returns</span>
                      <span className="font-bold text-white">& Orders</span>
                    </div>
                    <button
                      onClick={handleLogout}
                      className="text-white text-sm hover:border border-white px-3 py-2 rounded"
                    >
                      Sign Out
                    </button>
                  </>
                ) : (
                  <button
                    onClick={() => setShowAuth(true)}
                    className="flex flex-col text-xs cursor-pointer hover:border border-white p-2 rounded"
                  >
                    <span className="text-gray-300">Hello, sign in</span>
                    <span className="font-bold text-white">Account & Lists</span>
                  </button>
                )}
                
                <div className="relative flex items-center cursor-pointer hover:border border-white p-2 rounded">
                  <ShoppingCart className="w-8 h-8 text-white" />
                  <span className="absolute -top-1 -right-1 bg-[#FF9900] text-white text-xs font-bold rounded-full w-5 h-5 flex items-center justify-center">0</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Secondary Nav - Updated with all 5 categories */}
        <div className="bg-[#232F3E]">
          <div className="max-w-[1500px] mx-auto px-4">
            <div className="flex items-center space-x-6 h-10 text-sm overflow-x-auto">
              <button className="flex items-center space-x-2 text-white hover:border border-white px-2 py-1 rounded whitespace-nowrap font-medium">
                <Menu className="w-4 h-4" />
                <span>All</span>
              </button>
              <a href="#" className="text-white hover:border border-white px-2 py-1 rounded whitespace-nowrap">Today's Deals</a>
              <a href="#" className="text-white hover:border border-white px-2 py-1 rounded whitespace-nowrap">Customer Service</a>
              <a href="#" className="text-white hover:border border-white px-2 py-1 rounded whitespace-nowrap">Registry</a>
              <a href="#" className="text-white hover:border border-white px-2 py-1 rounded whitespace-nowrap">Gift Cards</a>
              
              {/* Cold Items Checkbox - Before Categories */}
              <label className="flex items-center space-x-2 text-white hover:bg-[#374151] px-3 py-1 rounded cursor-pointer whitespace-nowrap">
                <input
                  type="checkbox"
                  checked={showColdItems}
                  onChange={(e) => setShowColdItems(e.target.checked)}
                  className="w-4 h-4 rounded border-gray-300 text-yellow-400 focus:ring-yellow-400 focus:ring-2 cursor-pointer"
                />
                <span className="text-sm font-medium">Show Cold Items Only</span>
              </label>
              
              <div className="border-l border-gray-500 h-6 mx-2"></div>
              
              {/* Display all categories in secondary nav */}
              {categories.map(cat => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`text-white hover:border border-white px-2 py-1 rounded whitespace-nowrap ${
                    selectedCategory === cat ? 'border border-white font-medium' : ''
                  }`}
                >
                  {getCategoryDisplayName(cat)}
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      {/* Auth Modal */}
      {showAuth && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-2xl w-full max-w-sm">
            <div className="p-6">
              <div className="text-center mb-6">
                <div className="text-3xl font-bold mb-1">
                  <span className="text-gray-900">amazon</span>
                  <span className="text-[#FF9900]">.ml</span>
                </div>
                <h2 className="text-2xl font-normal text-gray-900 mt-4">
                  {isRegister ? 'Create account' : 'Sign in'}
                </h2>
              </div>

              {error && (
                <div className={`mb-4 p-3 rounded text-sm ${
                  error.includes('successful') 
                    ? 'bg-green-50 border border-green-300 text-green-800'
                    : 'bg-red-50 border border-red-300 text-red-800'
                }`}>
                  {error}
                </div>
              )}

              <form onSubmit={handleAuth} className="space-y-4">
                <div>
                  <label className="block text-sm font-bold text-gray-900 mb-1">
                    {isRegister ? 'Your name' : 'Email or mobile phone number'}
                  </label>
                  <input
                    type="text"
                    value={authForm.username}
                    onChange={(e) => setAuthForm({...authForm, username: e.target.value})}
                    className="w-full px-3 py-1.5 border border-gray-400 rounded-sm focus:border-[#e77600] focus:shadow-[0_0_3px_2px_rgb(228,121,17,0.5)] outline-none text-sm"
                    required
                  />
                </div>

                {isRegister && (
                  <div>
                    <label className="block text-sm font-bold text-gray-900 mb-1">Email</label>
                    <input
                      type="email"
                      value={authForm.email}
                      onChange={(e) => setAuthForm({...authForm, email: e.target.value})}
                      className="w-full px-3 py-1.5 border border-gray-400 rounded-sm focus:border-[#e77600] focus:shadow-[0_0_3px_2px_rgb(228,121,17,0.5)] outline-none text-sm"
                      required
                    />
                  </div>
                )}

                <div>
                  <label className="block text-sm font-bold text-gray-900 mb-1">Password</label>
                  <input
                    type="password"
                    value={authForm.password}
                    onChange={(e) => setAuthForm({...authForm, password: e.target.value})}
                    className="w-full px-3 py-1.5 border border-gray-400 rounded-sm focus:border-[#e77600] focus:shadow-[0_0_3px_2px_rgb(228,121,17,0.5)] outline-none text-sm"
                    required
                  />
                  {!isRegister && (
                    <p className="text-xs text-gray-600 mt-1">
                      Passwords must be at least 6 characters.
                    </p>
                  )}
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-[#FFD814] hover:bg-[#F7CA00] text-gray-900 font-medium py-2 rounded-lg border border-[#FCD200] shadow-sm disabled:opacity-50 transition text-sm"
                >
                  {loading ? 'Processing...' : (isRegister ? 'Create your Amazon account' : 'Sign in')}
                </button>

                <div className="text-xs text-gray-600 leading-relaxed">
                  By {isRegister ? 'creating an account' : 'continuing'}, you agree to Amazon's{' '}
                  <a href="#" className="text-blue-600 hover:text-[#C45500] hover:underline">Conditions of Use</a> and{' '}
                  <a href="#" className="text-blue-600 hover:text-[#C45500] hover:underline">Privacy Notice</a>.
                </div>
              </form>

              <div className="relative my-6">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-300"></div>
                </div>
                <div className="relative flex justify-center text-xs">
                  <span className="bg-white px-4 text-gray-500">New to Amazon?</span>
                </div>
              </div>

              <button
                onClick={() => {
                  setIsRegister(!isRegister);
                  setError('');
                }}
                className="w-full border border-gray-400 hover:bg-gray-50 text-gray-900 font-normal py-2 rounded-lg text-sm transition"
              >
                {isRegister ? 'Already have an account? Sign in' : 'Create your Amazon account'}
              </button>

              <button
                onClick={() => setShowAuth(false)}
                className="w-full mt-3 text-sm text-blue-600 hover:text-[#C45500] hover:underline"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-[1500px] mx-auto px-4 py-5">
        {/* Breadcrumb & Info Banner */}
        <div className="mb-5">
          <div className="text-xs text-gray-600 mb-3">
            <a href="#" className="hover:text-[#C45500] hover:underline">Home</a>
            <span className="mx-2">›</span>
            <a href="#" className="hover:text-[#C45500] hover:underline">{getCategoryDisplayName(selectedCategory)}</a>
            <span className="mx-2">›</span>
            <span className="text-gray-900 font-medium">Recommended for You</span>
          </div>
          
          {strategy && (
            <div className="space-y-3">
              {/* User Scenario Badge */}
              {userScenario && (
                <div className="inline-block" style={{
                  backgroundColor: userScenario.color,
                  padding: '12px 16px',
                  borderRadius: '8px',
                  color: '#fff'
                }}>
                  <div className="flex items-center space-x-2">
                    <span style={{ fontSize: '24px' }}>{userScenario.emoji}</span>
                    <div>
                      <div className="font-bold text-base">{userScenario.type}</div>
                      <div className="text-sm opacity-90">{userScenario.description}</div>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Algorithm Strategy */}
              <div className="bg-blue-50 border-l-4 border-blue-400 p-3 rounded-r">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900">
                      {isLoggedIn ? (
                        <>🎯 Algorithm Strategy: <span className="text-blue-700 font-bold">{strategy}</span></>
                      ) : (
                        <>Trending products in {getCategoryDisplayName(selectedCategory)} <span className="text-blue-700 font-bold">(Sign in for personalized recommendations)</span></>
                      )}
                    </p>
                    <p className="text-xs text-gray-600 mt-1">
                      {isLoggedIn 
                        ? 'Combining multiple algorithms based on your profile'
                        : 'Popular products based on ratings and recent activity'
                      }
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {loading && (
          <div className="flex justify-center items-center py-32">
            <div className="text-center">
              <div className="inline-block w-16 h-16 border-4 border-[#FF9900] border-t-transparent rounded-full animate-spin"></div>
              <p className="mt-6 text-gray-700 font-medium">Loading recommendations...</p>
            </div>
          </div>
        )}

        {error && !loading && (
          <div className="bg-red-50 border border-red-200 p-4 rounded mb-6">
            <p className="text-red-800 text-sm font-medium">{error}</p>
          </div>
        )}

        {!loading && recommendations.length === 0 && (
          <div className="text-center py-24 bg-white rounded-lg border border-gray-200 shadow-sm">
            <ShoppingCart className="w-24 h-24 text-gray-300 mx-auto mb-6" />
            {!isLoggedIn ? (
              <>
                <h2 className="text-3xl font-medium text-gray-900 mb-3">See personalized recommendations</h2>
                <p className="text-gray-600 mb-8 max-w-md mx-auto">
                  Sign in to get product recommendations based on your browsing history, ratings, and preferences
                </p>
                <button
                  onClick={() => setShowAuth(true)}
                  className="bg-[#FFD814] hover:bg-[#F7CA00] text-gray-900 font-medium px-12 py-2.5 rounded-lg border border-[#FCD200] shadow-sm text-sm"
                >
                  Sign in
                </button>
                <p className="text-sm text-gray-600 mt-4">
                  New customer?{' '}
                  <button
                    onClick={() => {
                      setShowAuth(true);
                      setIsRegister(true);
                    }}
                    className="text-blue-600 hover:text-[#C45500] hover:underline"
                  >
                    Start here.
                  </button>
                </p>
              </>
            ) : (
              <>
                <h2 className="text-2xl font-medium text-gray-900 mb-3">No recommendations found</h2>
                <p className="text-gray-500 text-sm">Rate some products to receive personalized suggestions</p>
              </>
            )}
          </div>
        )}

        {showColdItems && Object.keys(groupedColdItems).length === 0 && !loading && (
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-r">
            <div className="flex items-start">
              <span className="text-3xl mr-4">🌱</span>
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-2">No Cold Items Found</h2>
                <p className="text-sm text-gray-700 mb-2">
                  The current category ({getCategoryDisplayName(selectedCategory)}) doesn't have any cold items (products with limited training data).
                </p>
                <p className="text-sm text-gray-600">
                  Try selecting a different category or uncheck "Show Cold Items Only" to see all recommendations.
                </p>
              </div>
            </div>
          </div>
        )}

        {showColdItems && Object.keys(groupedColdItems).length > 0 && (
          <div>
            <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-r mb-6">
              <div className="flex items-start">
                <span className="text-2xl mr-3">🌱</span>
                <div>
                  <h2 className="text-lg font-bold text-gray-900 mb-1">Cold Items - By Training Rating Count</h2>
                  <p className="text-sm text-gray-700">Items grouped by number of ratings in training set. Scroll right to see more items in each row.</p>
                </div>
              </div>
            </div>
            
            {/* 4 Rows: 4-rating, 3-rating, 2-rating, 1-rating */}
            {[4, 3, 2, 1].map(ratingLevel => (
              <ColdItemRow
                key={ratingLevel}
                ratingLevel={ratingLevel}
                items={groupedColdItems[ratingLevel] || []}
                onRate={handleRate}
                isLoggedIn={isLoggedIn}
                setShowAuth={setShowAuth}
              />
            ))}
          </div>
        )}

        {!showColdItems && recommendations.length > 0 && (
          <div>
            <h1 className="text-2xl font-medium text-gray-900 mb-4">
              {isLoggedIn ? 'Recommended for you' : 'Trending products'} in {getCategoryDisplayName(selectedCategory)}
            </h1>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {recommendations.map((product, idx) => (
                <ProductCard
                  key={product.parent_asin}
                  product={product}
                  onRate={handleRate}
                  rank={idx + 1}
                  isLoggedIn={isLoggedIn}
                />
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-[#232F3E] text-white mt-16">
        <div className="bg-[#37475A] hover:bg-[#485769] cursor-pointer transition">
          <div className="max-w-[1500px] mx-auto px-4 py-4 text-center text-sm font-medium">
            Back to top
          </div>
        </div>
        <div className="max-w-[1500px] mx-auto px-4 py-10">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-sm">
            <div>
              <h3 className="font-bold mb-3">Get to Know Us</h3>
              <ul className="space-y-2 text-gray-300">
                <li><a href="#" className="hover:underline">About APRS</a></li>
                <li><a href="#" className="hover:underline">ML Models</a></li>
                <li><a href="#" className="hover:underline">Research Paper</a></li>
              </ul>
            </div>
            <div>
              <h3 className="font-bold mb-3">Recommendation System</h3>
              <ul className="space-y-2 text-gray-300">
                <li><a href="#" className="hover:underline">User-Based CF</a></li>
                <li><a href="#" className="hover:underline">Item-Based CF</a></li>
                <li><a href="#" className="hover:underline">Content-Based</a></li>
                <li><a href="#" className="hover:underline">Model-Based</a></li>
                <li><a href="#" className="hover:underline">Trending-Based</a></li>
                <li><a href="#" className="hover:underline">Hybrid Model</a></li>
              </ul>
            </div>
            <div>
              <h3 className="font-bold mb-3">Categories</h3>
              <ul className="space-y-2 text-gray-300">
                {categories.map(cat => (
                  <li key={cat}>
                    <button 
                      onClick={() => setSelectedCategory(cat)}
                      className="hover:underline text-left"
                    >
                      {getCategoryDisplayName(cat)}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="font-bold mb-3">Project Info</h3>
              <ul className="space-y-2 text-gray-300">
                <li>Northeastern University</li>
                <li>IE7275 Data Mining - Fall 2025</li>
                <li>Group 6</li>
                <li>Quoc Hung Le & Matthew Eckert</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="bg-[#131921] py-4">
          <div className="max-w-[1500px] mx-auto px-4 text-center text-xs text-gray-400">
            <p>Amazon Product Recommendation System © 2024 | Quoc Hung Le & Matthew Eckert</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function ProductCard({ product, onRate, rank, isLoggedIn }) {
  const [showRating, setShowRating] = useState(false);
  const [selectedRating, setSelectedRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [isWishlisted, setIsWishlisted] = useState(false);
  
  const itemScenario = product.item_scenario;

  const handleRatingSubmit = () => {
    if (selectedRating > 0) {
      onRate(product.parent_asin, selectedRating);
      setShowRating(false);
      setSelectedRating(0);
      setHoverRating(0);
    }
  };

  const formatPrice = (price) => {
    if (!price || price === 'N/A') return null;
    const priceStr = typeof price === 'string' ? price.replace(/[^0-9.]/g, '') : String(price);
    const priceNum = parseFloat(priceStr);
    return isNaN(priceNum) ? null : priceNum.toFixed(2);
  };

  const priceFormatted = formatPrice(product.price);
  const priceDollars = priceFormatted ? priceFormatted.split('.')[0] : null;
  const priceCents = priceFormatted ? priceFormatted.split('.')[1] : null;

  return (
    <div className="bg-white rounded border border-transparent hover:border-gray-300 hover:shadow-lg transition-all duration-200 overflow-hidden group cursor-pointer">
      <div className="relative bg-white p-4 flex items-center justify-center overflow-hidden" style={{ height: '256px' }}>
        {/* Item Scenario Badge */}
        {itemScenario && (
          <div className="absolute top-3 left-3 z-10">
            <div style={{
              backgroundColor: itemScenario.color,
              color: '#fff',
              fontSize: '11px',
              fontWeight: 'bold',
              padding: '4px 8px',
              borderRadius: '4px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
            }}>
              {itemScenario.emoji} {itemScenario.type}
            </div>
          </div>
        )}
        
        {rank <= 3 && !itemScenario && (
          <div className="absolute top-3 left-3 z-10">
            <div className="bg-[#CC0C39] text-white text-xs font-bold px-2.5 py-1 rounded-sm shadow">
              Best Seller #{rank}
            </div>
          </div>
        )}
        
        <button
          onClick={(e) => {
            e.stopPropagation();
            setIsWishlisted(!isWishlisted);
          }}
          className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity z-10"
        >
          <Heart 
            className={`w-6 h-6 ${isWishlisted ? 'fill-red-500 text-red-500' : 'text-gray-600'} hover:scale-110 transition-transform`}
          />
        </button>

        <img
          src={product.image_url || 'https://m.media-amazon.com/images/G/01/x-locale/common/grey-pixel.jpg'}
          alt={product.title}
          className="max-w-full max-h-full object-contain group-hover:scale-105 transition-transform duration-300"
          onError={(e) => { 
            e.target.src = 'https://m.media-amazon.com/images/G/01/x-locale/common/grey-pixel.jpg';
          }}
        />
      </div>

      <div className="p-3 space-y-2">
        <div className="text-xs text-gray-500">APRS sponsored</div>

        <h3 className="text-sm leading-tight text-gray-900 line-clamp-2 h-10 hover:text-[#C45500] cursor-pointer">
          {product.title || 'Product Name Unavailable'}
        </h3>

        <div className="flex items-center space-x-2">
          <div className="flex items-center">
            {[...Array(5)].map((_, i) => (
              <Star
                key={i}
                className={`w-3.5 h-3.5 ${
                  i < Math.floor(product.rating || 0)
                    ? 'fill-[#FFA41C] text-[#FFA41C]'
                    : 'fill-gray-200 text-gray-200'
                }`}
              />
            ))}
          </div>
          <span className="text-sm text-[#007185] hover:text-[#C45500] cursor-pointer">
            {product.rating_number ? `${product.rating_number.toLocaleString()}` : '0'}
          </span>
        </div>

        {priceDollars ? (
          <div className="flex items-baseline space-x-1">
            <span className="text-xs text-gray-900 align-top">$</span>
            <span className="text-2xl font-medium text-gray-900">{priceDollars}</span>
            <span className="text-sm text-gray-900">{priceCents}</span>
          </div>
        ) : (
          <div className="text-gray-600 text-sm">Price not available</div>
        )}

        <div className="flex items-center space-x-2 text-xs">
          <div className="bg-[#00A8E1] text-white font-bold px-1.5 py-0.5 rounded">prime</div>
          <span className="text-gray-700">FREE delivery</span>
        </div>

        <div className="flex items-center justify-between flex-wrap gap-1">
          <div className="text-xs bg-green-100 text-green-800 font-medium px-2 py-1 rounded">
            {Math.min(((product.score / 5) * 100), 100).toFixed(0)}% Match
          </div>
          
          {/* Show user's own rating if exists */}
          {product.user_rated && (
            <div className="text-xs bg-blue-100 text-blue-800 font-bold px-2 py-1 rounded flex items-center gap-1">
              <Star className="w-3 h-3 fill-blue-600" />
              {product.user_rating}★
            </div>
          )}
          
          {product.rating && !product.user_rated && (
            <div className="text-xs text-gray-600">
              {product.rating.toFixed(1)} ★
            </div>
          )}
        </div>

        {isLoggedIn && !showRating && (
          <div className="space-y-2 pt-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowRating(true);
              }}
              className="w-full bg-[#FFD814] hover:bg-[#F7CA00] text-gray-900 text-xs font-medium py-1.5 rounded-md border border-[#FCD200] shadow-sm transition truncate"
            >
              Rate Product
            </button>
            <button className="w-full bg-[#FFA41C] hover:bg-[#FA8900] text-white text-xs font-medium py-1.5 rounded-md shadow-sm transition truncate">
              Add to Cart
            </button>
          </div>
        )}

        {!isLoggedIn && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowAuth(true);
            }}
            className="w-full bg-[#FFD814] hover:bg-[#F7CA00] text-gray-900 text-sm font-medium py-2 rounded-lg border border-[#FCD200] shadow-sm transition"
          >
            Add to Cart
          </button>
        )}

        {isLoggedIn && showRating && (
          <div className="space-y-2 bg-gray-50 p-3 rounded-lg border border-gray-200">
            <p className="text-xs text-gray-700 font-medium text-center">How would you rate this?</p>
            <div className="flex justify-center space-x-1">
              {[1, 2, 3, 4, 5].map(rating => (
                <button
                  key={rating}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedRating(rating);
                  }}
                  onMouseEnter={() => setHoverRating(rating)}
                  onMouseLeave={() => setHoverRating(0)}
                  className="transition-transform hover:scale-110"
                >
                  <Star
                    className={`w-7 h-7 ${
                      rating <= (hoverRating || selectedRating)
                        ? 'fill-[#FFA41C] text-[#FFA41C]'
                        : 'fill-gray-200 text-gray-200'
                    }`}
                  />
                </button>
              ))}
            </div>
            <div className="flex space-x-2 pt-1">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRatingSubmit();
                }}
                disabled={selectedRating === 0}
                className="flex-1 bg-[#FFD814] hover:bg-[#F7CA00] text-gray-900 py-1.5 rounded text-sm font-medium border border-[#FCD200] disabled:opacity-50"
              >
                Submit
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowRating(false);
                  setSelectedRating(0);
                  setHoverRating(0);
                }}
                className="flex-1 bg-white border border-gray-300 text-gray-700 py-1.5 rounded text-sm hover:bg-gray-50"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ColdItemRow({ ratingLevel, items, onRate, isLoggedIn, setShowAuth }) {
  const scrollRef = React.useRef(null);
  const [currentPage, setCurrentPage] = useState(0);
  const itemsPerPage = 5;
  const totalPages = Math.ceil(items.length / itemsPerPage);

  const scroll = (direction) => {
    if (direction === 'left' && currentPage > 0) {
      setCurrentPage(currentPage - 1);
    } else if (direction === 'right' && currentPage < totalPages - 1) {
      setCurrentPage(currentPage + 1);
    }
  };

  if (items.length === 0) return null;

  const startIdx = currentPage * itemsPerPage;
  const endIdx = Math.min(startIdx + itemsPerPage, items.length);
  const displayedItems = items.slice(startIdx, endIdx);

  return (
    <div className="mb-8">
      <h3 className="text-lg font-bold text-gray-900 mb-3">
        {ratingLevel} Rating{ratingLevel > 1 ? 's' : ''}
        <span className="text-sm font-normal text-gray-600 ml-2">
          (page {currentPage + 1} of {totalPages} - showing {displayedItems.length} of {items.length} items)
        </span>
      </h3>

      <div className="relative">
        {/* Left Arrow */}
        {currentPage > 0 && (
          <button
            onClick={() => scroll('left')}
            className="absolute left-0 top-1/2 -translate-y-1/2 z-20 bg-white hover:bg-gray-100 shadow-2xl rounded-full transition-all hover:scale-110"
            style={{ width: '56px', height: '56px', border: '2px solid #232F3E' }}
          >
            <ChevronDown className="w-8 h-8 rotate-90 text-gray-900 mx-auto" />
          </button>
        )}

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 px-16">
          {displayedItems.map(item => (
            <ColdItemCard key={item.parent_asin} item={item} onRate={onRate} isLoggedIn={isLoggedIn} setShowAuth={setShowAuth} />
          ))}
        </div>

        {/* Right Arrow */}
        {currentPage < totalPages - 1 && (
          <button
            onClick={() => scroll('right')}
            className="absolute right-0 top-1/2 -translate-y-1/2 z-20 bg-white hover:bg-gray-100 shadow-2xl rounded-full transition-all hover:scale-110"
            style={{ width: '56px', height: '56px', border: '2px solid #232F3E' }}
          >
            <ChevronDown className="w-8 h-8 -rotate-90 text-gray-900 mx-auto" />
          </button>
        )}
      </div>
    </div>
  );
}

function ColdItemCard({ item, onRate, isLoggedIn, setShowAuth }) {
  const [showRating, setShowRating] = useState(false);
  const [selectedRating, setSelectedRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [isWishlisted, setIsWishlisted] = useState(false);

  const handleRatingSubmit = () => {
    if (selectedRating > 0) {
      onRate(item.parent_asin, selectedRating);
      setShowRating(false);
      setSelectedRating(0);
      setHoverRating(0);
    }
  };

  const formatPrice = (price) => {
    if (!price || price === 'N/A') return null;
    const priceStr = typeof price === 'string' ? price.replace(/[^0-9.]/g, '') : String(price);
    const priceNum = parseFloat(priceStr);
    return isNaN(priceNum) ? null : priceNum.toFixed(2);
  };

  const priceFormatted = formatPrice(item.price);
  const priceDollars = priceFormatted ? priceFormatted.split('.')[0] : null;
  const priceCents = priceFormatted ? priceFormatted.split('.')[1] : null;

  return (
    <div className="bg-white rounded border border-transparent hover:border-gray-300 hover:shadow-lg transition-all duration-200 overflow-hidden group cursor-pointer">
      <div className="relative bg-white p-4 flex items-center justify-center overflow-hidden" style={{ height: '256px' }}>
        <button
          onClick={(e) => {
            e.stopPropagation();
            setIsWishlisted(!isWishlisted);
          }}
          className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity z-10"
        >
          <Heart 
            className={`w-6 h-6 ${isWishlisted ? 'fill-red-500 text-red-500' : 'text-gray-600'} hover:scale-110 transition-transform`}
          />
        </button>

        <img
          src={item.image_url || 'https://m.media-amazon.com/images/G/01/x-locale/common/grey-pixel.jpg'}
          alt={item.title}
          className="max-w-full max-h-full object-contain group-hover:scale-105 transition-transform duration-300"
          onError={(e) => { 
            e.target.src = 'https://m.media-amazon.com/images/G/01/x-locale/common/grey-pixel.jpg';
          }}
        />
      </div>

      <div className="p-3 space-y-2">
        <div className="text-xs text-gray-500">APRS sponsored</div>

        <h3 className="text-sm leading-tight text-gray-900 line-clamp-2 h-10 hover:text-[#C45500] cursor-pointer">
          {item.title || 'Product Name Unavailable'}
        </h3>

        {item.rating > 0 && (
          <div className="flex items-center space-x-2">
            <div className="flex items-center">
              {[...Array(5)].map((_, i) => (
                <Star
                  key={i}
                  className={`w-3.5 h-3.5 ${
                    i < Math.floor(item.rating)
                      ? 'fill-[#FFA41C] text-[#FFA41C]'
                      : 'fill-gray-200 text-gray-200'
                  }`}
                />
              ))}
            </div>
            <span className="text-sm text-[#007185] hover:text-[#C45500] cursor-pointer">
              {item.rating_number ? `${item.rating_number.toLocaleString()}` : item.rating.toFixed(1)}
            </span>
          </div>
        )}

        {priceDollars ? (
          <div className="flex items-baseline space-x-1">
            <span className="text-xs text-gray-900 align-top">$</span>
            <span className="text-2xl font-medium text-gray-900">{priceDollars}</span>
            <span className="text-sm text-gray-900">{priceCents}</span>
          </div>
        ) : (
          <div className="text-gray-600 text-sm">Price not available</div>
        )}

        <div className="flex items-center space-x-2 text-xs">
          <div className="bg-[#00A8E1] text-white font-bold px-1.5 py-0.5 rounded">prime</div>
          <span className="text-gray-700">FREE delivery</span>
        </div>

        {isLoggedIn && !showRating && (
          <div className="space-y-2 pt-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowRating(true);
              }}
              className="w-full bg-[#FFD814] hover:bg-[#F7CA00] text-gray-900 text-xs font-medium py-1.5 rounded-md border border-[#FCD200] shadow-sm transition truncate"
            >
              Rate Product
            </button>
            <button className="w-full bg-[#FFA41C] hover:bg-[#FA8900] text-white text-xs font-medium py-1.5 rounded-md shadow-sm transition truncate">
              Add to Cart
            </button>
          </div>
        )}

        {!isLoggedIn && (
          <div className="space-y-2 pt-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowAuth(true);
              }}
              className="w-full bg-[#FFD814] hover:bg-[#F7CA00] text-gray-900 text-xs font-medium py-1.5 rounded-md border border-[#FCD200] shadow-sm transition truncate"
            >
              Rate Product
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowAuth(true);
              }}
              className="w-full bg-[#FFA41C] hover:bg-[#FA8900] text-white text-xs font-medium py-1.5 rounded-md shadow-sm transition truncate"
            >
              Add to Cart
            </button>
          </div>
        )}

        {isLoggedIn && showRating && (
          <div className="space-y-2 bg-gray-50 p-3 rounded-lg border border-gray-200">
            <p className="text-xs text-gray-700 font-medium text-center">How would you rate this?</p>
            <div className="flex justify-center space-x-1">
              {[1, 2, 3, 4, 5].map(rating => (
                <button
                  key={rating}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedRating(rating);
                  }}
                  onMouseEnter={() => setHoverRating(rating)}
                  onMouseLeave={() => setHoverRating(0)}
                  className="transition-transform hover:scale-110"
                >
                  <Star
                    className={`w-7 h-7 ${
                      rating <= (hoverRating || selectedRating)
                        ? 'fill-[#FFA41C] text-[#FFA41C]'
                        : 'fill-gray-200 text-gray-200'
                    }`}
                  />
                </button>
              ))}
            </div>
            <div className="flex space-x-2 pt-1">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRatingSubmit();
                }}
                disabled={selectedRating === 0}
                className="flex-1 bg-[#FFD814] hover:bg-[#F7CA00] text-gray-900 py-1.5 rounded text-sm font-medium border border-[#FCD200] disabled:opacity-50"
              >
                Submit
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowRating(false);
                  setSelectedRating(0);
                  setHoverRating(0);
                }}
                className="flex-1 bg-white border border-gray-300 text-gray-700 py-1.5 rounded text-sm hover:bg-gray-50"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;