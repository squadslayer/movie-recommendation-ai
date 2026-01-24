# Design Document: Authentication & Onboarding Flow Completion

## Overview

This design completes the authentication and onboarding flow for a movie recommendation app by implementing route guards, personalized content display, user interaction tracking, and enhanced user experience features. The solution builds upon existing infrastructure including Supabase authentication, backend APIs, and frontend components to create a seamless user journey.

The design follows a layered architecture approach:
- **Presentation Layer**: React components with route guards and personalized UI
- **Business Logic Layer**: Authentication state management and user interaction tracking
- **Data Layer**: Integration with existing APIs and database logging
- **Caching Layer**: Leveraging existing 10-minute TTL caching strategy

## Architecture

The system architecture extends the existing movie recommendation app with four main integration points:

```mermaid
graph TB
    subgraph "Frontend Layer"
        RG[Route Guard]
        HP[Home Page]
        NAV[Navigation Bar]
        AUTH[Auth Context]
    end
    
    subgraph "Business Logic"
        IL[Interaction Logger]
        PS[Personalization Service]
    end
    
    subgraph "API Layer"
        TA[/api/trending]
        PA[/api/profile/*]
    end
    
    subgraph "Data Layer"
        SP[Supabase Auth]
        UP[user_profile]
        IE[user_interaction_event]
        CACHE[In-memory Cache]
    end
    
    RG --> AUTH
    RG --> PA
    HP --> TA
    HP --> PS
    NAV --> AUTH
    IL --> IE
    PS --> UP
    TA --> CACHE
    AUTH --> SP
```

**Key Architectural Decisions:**

1. **Centralized Route Protection**: Single Route Guard component handles all authentication-based routing decisions
2. **Existing API Integration**: Leverages current /api/trending and /api/profile endpoints without modifications
3. **Event-Driven Interaction Logging**: Asynchronous logging system that doesn't block user interactions
4. **Progressive Enhancement**: Graceful degradation when APIs fail, with cached data fallbacks using in-memory process-level caching

## Components and Interfaces

### Route Guard Component

**Purpose**: Centralized authentication and routing logic
**Location**: Wraps main application routes

```typescript
interface RouteGuardProps {
  children: React.ReactNode;
  requireAuth?: boolean;
  requireOnboarding?: boolean;
}

interface AuthState {
  user: User | null;
  profile: UserProfile | null;
  loading: boolean;
}
```

**Behavior**:
- Checks authentication state from AuthContext
- Fetches user profile data when authenticated
- Implements redirect logic based on auth/onboarding status
- Preserves intended destination URLs for post-auth redirect

### Personalized Home Page

**Purpose**: Display trending and personalized movie content
**Location**: /home route (protected)

```typescript
interface HomePageState {
  trendingMovies: Movie[];
  personalizedSections: PersonalizedSection[];
  loading: boolean;
  error: string | null;
}

interface PersonalizedSection {
  title: string;
  movies: Movie[];
  type: 'trending' | 'language' | 'genre';
}
```

**Behavior**:
- Fetches trending data from /api/trending
- Organizes content into personalized sections based on user preferences
- Handles loading states and error conditions
- Implements fallback to cached data when APIs fail

### Interaction Logger Service

**Purpose**: Track user engagement events
**Location**: Utility service used across components

```typescript
interface InteractionEvent {
  userProfileId: string;  // References domain user, not auth user
  movieId: string;
  interactionType: 'click' | 'view' | 'onboarding_complete';
  timestamp: Date;
  metadata?: Record<string, any>;
}

interface InteractionLogger {
  logMovieClick(movieId: string): Promise<void>;
  logOnboardingComplete(): Promise<void>;
  logMovieView(movieId: string): Promise<void>;
}
```

**Behavior**:
- Queues interaction events to prevent database overload
- Handles authentication state (no logging for anonymous users)
- Graceful error handling that doesn't disrupt user experience
- Batch processing for high-frequency events

### Enhanced Navigation Bar

**Purpose**: Display user authentication state and controls
**Location**: Global navigation component

```typescript
interface NavBarProps {
  user: User | null;
  profile: UserProfile | null;
}

interface UserDropdownState {
  isOpen: boolean;
  avatar: string | null;
}
```

**Behavior**:
- Shows user avatar and dropdown when authenticated
- Provides logout functionality
- Displays loading states during authentication transitions

## Data Models

### Data Schema Integration

The system leverages the existing step-based enterprise schema:
- **user_profile**: Domain identity and onboarding state
- **user_preferences**: JSONB storage for languages and genres  
- **user_interaction_event**: Immutable interaction log with domain user references

No schema changes are introduced by this design. The system works with the established enterprise data model where preferences are stored in JSONB format and interactions reference domain users rather than auth users.

### API Response Models

**Trending API Response** (`/api/trending`):
```typescript
interface TrendingResponse {
  movies: Movie[];
  cached_at: string;
  expires_at: string;
}

interface Movie {
  id: string;
  title: string;
  poster_path: string;
  overview: string;
  release_date: string;
  genre_ids: number[];
  original_language: string;
  vote_average: number;
}
```

**Profile API Response** (`/api/profile/onboard`):
```typescript
interface ProfileResponse {
  profile: UserProfile;
  success: boolean;
}

interface UserProfile {
  id: string;
  username: string;
  preferred_languages: string[];
  favorite_genres: string[];
  onboarding_completed: boolean;
}
```

Now I need to use the prework tool to analyze the acceptance criteria before writing the Correctness Properties section:

<function_calls>
<invoke name="prework">
<parameter name="featureName">auth-flow-completion

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Authentication-based routing decisions
*For any* user authentication state and onboarding status, the Route Guard must redirect users to the appropriate destination: unauthenticated users to login, authenticated users without onboarding to the wizard, and authenticated users with completed onboarding to their intended destination
**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

### Property 2: URL preservation during authentication redirects
*For any* protected route accessed by an unauthenticated user, the original URL must be preserved and used for redirect after successful authentication
**Validates: Requirements 1.5**

### Property 3: Trending data display for authenticated users
*For any* authenticated user visiting the home page, trending movies from the API must be displayed in the interface
**Validates: Requirements 2.1**

### Property 4: Personalized content organization
*For any* user with specified preferences, the home page must organize movies into personalized sections that match their language and genre preferences
**Validates: Requirements 2.2, 2.3, 2.4**

### Property 5: Cache TTL consistency
*For any* trending data request within the 10-minute TTL window, cached results must be returned, and after expiration, fresh data must be fetched and cached
**Validates: Requirements 2.5, 5.1, 5.2**

### Property 6: Movie interaction logging
*For any* authenticated user clicking on a movie card, an interaction event must be recorded in the user_interaction_event table with all required fields
**Validates: Requirements 3.1, 3.3**

### Property 7: Onboarding completion logging
*For any* user completing the onboarding process, an onboarding completion event must be logged to the interaction system
**Validates: Requirements 3.2**

### Property 8: Anonymous user interaction exclusion
*For any* anonymous user interaction with movie content, no interaction events shall be generated or logged
**Validates: Requirements 3.6**

### Property 9: Interaction logging error resilience
*For any* interaction logging failure, the user experience must continue normally without disruption or error display
**Validates: Requirements 3.4**

### Property 10: Rapid interaction queuing
*For any* sequence of rapid user interactions, events must be queued properly to prevent database overload while ensuring all interactions are eventually logged
**Validates: Requirements 3.5**

### Property 11: Loading state display
*For any* trending data loading operation, appropriate loading indicators must be displayed to users during the fetch process
**Validates: Requirements 4.1**

### Property 12: API failure error handling
*For any* API request failure, user-friendly error messages must be displayed without exposing technical implementation details
**Validates: Requirements 4.2, 4.5**

### Property 13: Cached data fallback
*For any* API failure where cached data is available, the cached content must be displayed with appropriate indicators showing it's cached data
**Validates: Requirements 4.3**

### Property 14: Authenticated user UI elements
*For any* authenticated user, the navigation bar must display user avatar and dropdown menu elements
**Validates: Requirements 4.4**

### Property 15: API endpoint consistency
*For any* trending data request, the system must use the existing /api/trending endpoint without modifications to maintain API contract consistency
**Validates: Requirements 5.3**

## System Invariants

The following invariants must hold throughout system operation:

1. **Profile-Preferences Consistency**: A user with onboarding_completed = true must always have a corresponding user_preferences record
2. **Immutable Event Log**: user_interaction_event is append-only and never mutated
3. **Deterministic Routing**: Routing decisions depend only on (auth_state, onboarding_completed) and are deterministic
4. **Event Ordering**: Events are eventually consistent; ordering across sessions is not guaranteed

## Error Handling

The system implements comprehensive error handling across all layers:

### Authentication Errors
- **Invalid credentials**: Display user-friendly messages without exposing security details
- **Session expiration**: Automatically redirect to login while preserving intended destination
- **Profile fetch failures**: Graceful degradation with retry mechanisms

### API Integration Errors
- **Trending API failures**: Fall back to cached data when available, display error states when cache is empty
- **Network timeouts**: Implement retry logic with exponential backoff
- **Rate limiting**: Queue requests and implement client-side throttling

### User Interaction Errors
- **Logging failures**: Continue user experience while queuing failed events for retry
- **Database connectivity issues**: Implement offline queuing with sync when connection restored
- **Validation errors**: Provide clear feedback for invalid user inputs

### UI Error States
- **Loading failures**: Display retry options and fallback content
- **Component errors**: Implement error boundaries to prevent application crashes
- **State inconsistencies**: Automatic state recovery and user notification

## Testing Strategy

The testing approach combines unit testing for specific scenarios with property-based testing for comprehensive coverage:

### Unit Testing Focus
- **Specific user flows**: Login → Onboarding → Home page navigation
- **Edge cases**: Empty preference lists, network failures, invalid user states
- **Integration points**: AuthContext integration, API response handling
- **Error conditions**: Authentication failures, API timeouts, logging errors

### Property-Based Testing Configuration
- **Testing library**: React Testing Library with Jest for React components, QuickCheck-style generators for data
- **Test iterations**: Minimum 100 iterations per property test to ensure comprehensive input coverage
- **Test tagging**: Each property test tagged with format: **Feature: auth-flow-completion, Property {number}: {property_text}**

### Property Test Implementation
Each correctness property will be implemented as a single property-based test:

1. **Authentication routing properties**: Generate random user states and verify routing decisions
2. **Content personalization properties**: Generate users with various preference combinations
3. **Interaction logging properties**: Generate interaction events and verify logging behavior
4. **Error handling properties**: Simulate various failure conditions and verify graceful handling
5. **Caching properties**: Test cache behavior across different time windows and failure scenarios

### Test Data Generation
- **User profiles**: Random combinations of authentication states, onboarding status, and preferences
- **Movie data**: Generated movie objects with various genres, languages, and metadata
- **Interaction events**: Random user interactions with different timing patterns
- **Error conditions**: Simulated API failures, network issues, and invalid states

The dual testing approach ensures both concrete bug detection through unit tests and general correctness verification through property-based testing, providing comprehensive coverage of the authentication and onboarding flow completion.